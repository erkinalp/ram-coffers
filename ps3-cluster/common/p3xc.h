/*
 * p3xc.h - PS3 eXpert Cluster wire protocol (C), byte-identical to protocol.py
 *
 * Frame (big-endian / network order), matching ps3_cluster/protocol.py:
 *   uint32 length (of everything after this field)
 *   char   magic[4] = "P3XC"
 *   uint8  version  = 1
 *   uint8  msg_type
 *   uint16 layer
 *   uint16 expert
 *   uint32 token_id
 *   uint8  dtype
 *   uint8  ndim
 *   uint32 shape[ndim]
 *   payload: product(shape) elements, big-endian, itemsize per dtype
 *
 * On the PS3 PPE (big-endian) the on-wire order equals host order, so the
 * float/int conversions below are no-ops; on a little-endian host-sim build
 * they byte-swap. Depends only on the C standard library + arpa/inet.
 */
#ifndef P3XC_H
#define P3XC_H

#include <stdint.h>
#include <string.h>
#include <arpa/inet.h>   /* htonl/ntohl/htons/ntohs */

#define P3XC_MAGIC   "P3XC"
#define P3XC_VERSION 1

#define P3XC_REQ  1
#define P3XC_RSP  2
#define P3XC_ERR  3
#define P3XC_PING 4
#define P3XC_PONG 5

#define P3XC_DT_F32 1
#define P3XC_DT_F16 2
#define P3XC_DT_U8  4

/* IANA-style fixed service port for the expert cluster (unassigned range). */
#define P3XC_DEFAULT_PORT 3830

typedef struct {
    uint8_t  msg_type;
    uint16_t layer;
    uint16_t expert;
    uint32_t token_id;
    uint8_t  dtype;
    uint8_t  ndim;
    uint32_t shape[4];
    uint32_t count;      /* product(shape) */
} p3xc_hdr_t;

/* Host<->big-endian float32. On BE PPE this is the identity. */
static inline uint32_t p3xc_f32_to_be(float f) {
    uint32_t u; memcpy(&u, &f, 4); return htonl(u);
}
static inline float p3xc_be_to_f32(uint32_t be) {
    uint32_t u = ntohl(be); float f; memcpy(&f, &u, 4); return f;
}

/* Parse a frame body (without the 4-byte length prefix). Returns payload
 * offset within body, or -1 on error. Fills hdr. */
static inline long p3xc_parse(const uint8_t *body, uint32_t body_len,
                              p3xc_hdr_t *hdr) {
    if (body_len < 16) return -1;
    if (memcmp(body, P3XC_MAGIC, 4) != 0) return -1;
    if (body[4] != P3XC_VERSION) return -1;
    hdr->msg_type = body[5];
    hdr->layer    = ntohs(*(const uint16_t *)(body + 6));
    hdr->expert   = ntohs(*(const uint16_t *)(body + 8));
    hdr->token_id = ntohl(*(const uint32_t *)(body + 10));
    hdr->dtype    = body[14];
    hdr->ndim     = body[15];
    if (hdr->ndim > 4) return -1;
    long off = 16;
    hdr->count = 1;
    for (uint8_t i = 0; i < hdr->ndim; ++i) {
        if ((uint32_t)(off + 4) > body_len) return -1;
        hdr->shape[i] = ntohl(*(const uint32_t *)(body + off));
        hdr->count *= hdr->shape[i];
        off += 4;
    }
    return off;
}

/* Serialise a float32 vector response into out (must hold >= 4+16+4*ndim+4*n).
 * Returns total bytes written including the 4-byte length prefix. */
static inline uint32_t p3xc_write_f32(uint8_t *out, uint8_t msg_type,
                                      uint16_t layer, uint16_t expert,
                                      uint32_t token_id,
                                      const float *data, uint32_t n) {
    uint8_t *p = out + 4;                 /* leave room for length prefix */
    memcpy(p, P3XC_MAGIC, 4); p += 4;
    *p++ = P3XC_VERSION;
    *p++ = msg_type;
    *(uint16_t *)p = htons(layer);  p += 2;
    *(uint16_t *)p = htons(expert); p += 2;
    *(uint32_t *)p = htonl(token_id); p += 4;
    *p++ = P3XC_DT_F32;
    *p++ = 1;                              /* ndim */
    *(uint32_t *)p = htonl(n); p += 4;     /* shape[0] */
    for (uint32_t i = 0; i < n; ++i) {
        *(uint32_t *)p = p3xc_f32_to_be(data[i]);
        p += 4;
    }
    uint32_t body_len = (uint32_t)(p - (out + 4));
    *(uint32_t *)out = htonl(body_len);
    return body_len + 4;
}

#endif /* P3XC_H */
