/*
 * Copyright (c) 2019, The Bifrost Authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "base.hpp"
#include <cmath>

#pragma pack(1)
struct cor_hdr_type {
	uint32_t sync_word;
	uint32_t frame_count_word;
	uint32_t second_count;
	uint16_t first_chan;
	uint16_t gain;
	uint64_t time_tag;
	uint32_t navg;
	uint16_t stand0;
	uint16_t stand1;
};

class CORDecoder : virtual public PacketDecoder {
   inline bool valid_packet(const PacketDesc* pkt) const {
	    return (pkt->sync       == 0x5CDEC0DE &&
	            pkt->src        >= 0 &&
               	pkt->src        <  _nsrc &&
	            pkt->time_tag   >= 0 &&
	            pkt->chan0      >= 0);
    }
public:
    CORDecoder(int nsrc, int src0) : PacketDecoder(nsrc, src0) {}
    inline bool operator()(const uint8_t* pkt_ptr,
	                       int            pkt_size,
	                       PacketDesc*    pkt) const {
	    if( pkt_size < (int)sizeof(cor_hdr_type) ) {
		    return false;
	    }
	    const cor_hdr_type* pkt_hdr  = (cor_hdr_type*)pkt_ptr;
	    const uint8_t*      pkt_pld  = pkt_ptr  + sizeof(cor_hdr_type);
	    int                 pld_size = pkt_size - sizeof(cor_hdr_type);
        // HACK
        uint8_t             nserver = 1;//(be32toh(pkt_hdr->frame_count_word) >> 8) & 0xFF;
        // HACK
	    uint8_t             server = 1;//be32toh(pkt_hdr->frame_count_word) & 0xFF;
	    uint16_t            stand0 = be16toh(pkt_hdr->stand0) - 1;
	    uint16_t            stand1 = be16toh(pkt_hdr->stand1) - 1;
	    uint16_t            nstand = (sqrt(8*_nsrc+1)-1)/2;
	    pkt->sync         = pkt_hdr->sync_word;
	    pkt->time_tag     = be64toh(pkt_hdr->time_tag);
	    pkt->decimation   = be32toh(pkt_hdr->navg);
	    pkt->seq          = pkt->time_tag / 196000000 / (pkt->decimation / 100);
	    pkt->nsrc         = _nsrc;
	    pkt->src          = stand0*(2*(nstand-1)+1-stand0)/2 + stand1 + 1 - _src0;
	    pkt->chan0        = be16toh(pkt_hdr->first_chan);
        pkt->nchan        = nserver * (pld_size/(8*4));     // Stores the total number of channels
	    pkt->tuning       = (nserver << 8) | (server - 1);  // Stores the number of servers and 
                                                            // the server that sent this packet
	    pkt->gain         = be16toh(pkt_hdr->gain);
	    pkt->payload_size = pld_size;
	    pkt->payload_ptr  = pkt_pld;
        /*
        if( stand0 == 0 || (stand0 == 1 && stand1 < 2) ) {
            std::cout << "nsrc:   " << pkt->nsrc << std::endl;
            std::cout << "stand0: " << stand0 << std::endl;
            std::cout << "stand1: " << stand1 << std::endl;
            std::cout << "src:    " << pkt->src << std::endl;
            std::cout << "chan0:  " << pkt->chan0 << std::endl;
            std::cout << "nchan:  " << pld_size/32 << std::endl;
            std::cout << "navg:   " << pkt->decimation << std::endl;
            std::cout << "tuning: " << pkt->tuning << std::endl;
            std::cout << "valid:  " << this->valid_packet(pkt) << std::endl;
        }
        */
        return this->valid_packet(pkt);
    }
};

class CORProcessor : virtual public PacketProcessor {
public:
    inline void operator()(const PacketDesc* pkt,
	                       uint64_t          seq0,
	                       uint64_t          nseq_per_obuf,
	                       int               nbuf,
	                       uint8_t*          obufs[],
	                       size_t            ngood_bytes[],
	                       size_t*           src_ngood_bytes[]) {
	    int    obuf_idx = ((pkt->seq - seq0 >= 1*nseq_per_obuf) +
	                       (pkt->seq - seq0 >= 2*nseq_per_obuf));
	    size_t obuf_seq0 = seq0 + obuf_idx*nseq_per_obuf;
	    size_t nbyte = pkt->payload_size;
	    ngood_bytes[obuf_idx]               += nbyte;
	    src_ngood_bytes[obuf_idx][pkt->src] += nbyte;
	    int payload_size = pkt->payload_size;
	
	    size_t obuf_offset = (pkt->seq-obuf_seq0)*pkt->nsrc*payload_size;
	    typedef aligned256_type itype;      // 32+32 * 4 pol. products
	    typedef aligned256_type otype;
	
	    // Note: Using these SSE types allows the compiler to use SSE instructions
	    //         However, they require aligned memory (otherwise segfault)
	    itype const* __restrict__ in  = (itype const*)pkt->payload_ptr;
	    otype*       __restrict__ out = (otype*      )&obufs[obuf_idx][obuf_offset];
	    
	    // Convinience
	    int bl = pkt->src;
	    //int nbl = pkt->nsrc;
	    int server = pkt->tuning & 0xFF;
	    int nserver = (pkt->tuning >> 8) & 0xFF;
        int nchan = pkt->nchan/nserver;
	    
	    int chan = 0;
	    for( ; chan<nchan; ++chan ) {
		    out[bl*nserver*nchan + server*nchan + chan] = in[chan];
	    }
    }
	
    inline void blank_out_source(uint8_t* data,
	                             int      src,  // bl
	                             int      nsrc, // nbl
	                             int      nchan,
	                             int      nseq) {
	    typedef aligned256_type otype;
	    otype* __restrict__ aligned_data = (otype*)data;
        for( int t=0; t<nseq; ++t ) {
		    ::memset(&aligned_data[t*nsrc*nchan + src*nchan],
			             0, nchan*sizeof(otype));
	    }
    }
};
