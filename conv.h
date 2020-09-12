#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

// common headers
#include <stdio.h>
#include <string.h>

using namespace hls;

#define cal_aligned_size(x,y) ((x+y-1)/y*y)

typedef ap_uint<192> U1_ConfigInst;

// Data types
typedef float U1_data_t0;
typedef ap_uint<512> U1_bus_t0;
#define U1_DATA0_WIDTH 32
#define U1_DATA0_PACK_FACTOR (512/U1_DATA0_WIDTH)
typedef float U1_data_t1;
typedef ap_uint<512> U1_bus_t1;
#define U1_DATA1_WIDTH 32
#define U1_DATA1_PACK_FACTOR (512/U1_DATA1_WIDTH)
typedef float U1_data_t2;
typedef ap_uint<512> U1_bus_t2;
#define U1_DATA2_WIDTH 32
#define U1_DATA2_PACK_FACTOR (512/U1_DATA2_WIDTH)
typedef unsigned int uint;
union ufloat{
	float f;
	unsigned int u;
};

// Macros
#define U1_IN_IMG_W_T 50
#define U1_OUT_NUM 960
#define U1_IN_IMG_H_T 14
#define U1_OUT_IMG_H 384
#define U1_LAYER_BATCH 2
#define U1_STRIDE 1
#define U1_K 3
#define U1_OUT_NUM_T 96
#define U1_IN_IMG_W 386
#define U1_IN_IMG_H 386
#define U1_IN_NUM_T 96
#define U1_IN_NUM 960
#define U1_OUT_IMG_H_T 12
#define U1_OUT_IMG_W_T 48
#define U1_OUT_IMG_W 384
#define U1_DATA0_SIZE 143036160
#define U1_DATA0_SIZE_ALIGNED (cal_aligned_size(143036160, U1_DATA0_PACK_FACTOR))
#define U1_DATA1_SIZE 8294400
#define U1_DATA1_SIZE_ALIGNED (cal_aligned_size(8294400, U1_DATA1_PACK_FACTOR))
#define U1_DATA2_SIZE 141557760
#define U1_DATA2_SIZE_ALIGNED (cal_aligned_size(141557760, U1_DATA2_PACK_FACTOR))

#define U1_ROW_IL_FACTOR 12
#define U1_COL_IL_FACTOR 6
#define U1_SA_ROWS 8
#define U1_SA_COLS 8
#define U1_LOCAL_REG_NUM 864
#define U1_LOCAL_ACCUM_NUM 108
#define U1_SIMD_FACTOR 8
#define U1_DATA0_FC_SIMD_FACTOR 8
#define U1_DATA0_FC_GROUP_FACTOR 1
#define U1_DATA0_FC_SPLIT_FACTOR 1
#define U1_DATA1_FC_SIMD_FACTOR 8
#define U1_DATA1_FC_GROUP_FACTOR 1
#define U1_DATA1_FC_SPLIT_FACTOR 1
#define U1_DATA2_FC_SIMD_FACTOR 8
#define U1_DATA2_FC_GROUP_FACTOR 1
#define U1_DATA2_FC_SPLIT_FACTOR 1

#define U1_DATA0_BUF_SIZE 10752
#define U1_DATA0_HEAD_BUF_SIZE 672000
#define U1_DATA1_BUF_SIZE 10368
#define U1_DATA1_HEAD_BUF_SIZE 829440
#define U1_DATA2_BUF_SIZE 6912
#define U1_DATA2_HEAD_BUF_SIZE 552960

// Functions and structs
struct U1_Data0TransferChannelType{
	U1_Data0TransferChannelType(){}
	U1_Data0TransferChannelType(
			ap_uint<U1_DATA0_WIDTH*U1_DATA0_FC_SIMD_FACTOR> data_t,
			unsigned int feeder_id_t,
			bool new_pair_t,
			bool last_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		feeder_id = feeder_id_t;
		new_pair = new_pair_t;
		last_pair = last_pair_t;
		FILTER_S = filter_s_t;
	}
	ap_uint<U1_DATA0_WIDTH*U1_DATA0_FC_SIMD_FACTOR> data;
	unsigned int feeder_id;
	bool new_pair;
	bool last_pair;
	unsigned int FILTER_S;
};

struct U1_Data1TransferChannelType{
	U1_Data1TransferChannelType(){}
	U1_Data1TransferChannelType(
			ap_uint<U1_DATA1_WIDTH*U1_DATA1_FC_SIMD_FACTOR> data_t,
			unsigned int feeder_id_t,
			bool new_pair_t,
			bool last_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		feeder_id = feeder_id_t;
		new_pair = new_pair_t;
		last_pair = last_pair_t;
		FILTER_S = filter_s_t;
	}
	ap_uint<U1_DATA1_WIDTH*U1_DATA1_FC_SIMD_FACTOR> data;
	unsigned int feeder_id;
	bool new_pair;
	bool last_pair;
	unsigned int FILTER_S;
};

struct U1_Data2TransferChannelType{
	U1_Data2TransferChannelType(){}
	U1_Data2TransferChannelType(
			ap_uint<U1_DATA2_WIDTH*U1_DATA2_FC_SIMD_FACTOR> data_t){
		data = data_t;
	}
	ap_uint<U1_DATA2_WIDTH*U1_DATA2_FC_SIMD_FACTOR> data;
};

struct U1_Data0PEChannelType{
	U1_Data0PEChannelType(){}
	U1_Data0PEChannelType(
			ap_uint<256> data_t
	){
		data = data_t;
	}
	U1_Data0PEChannelType(
			ap_uint<256> data_t,
			bool new_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		new_pair = new_pair_t;
		FILTER_S = filter_s_t;
	}
	U1_Data0PEChannelType(
			ap_uint<256> data_t,
			bool new_pair_t,
			bool last_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		new_pair = new_pair_t;
		last_pair = last_pair_t;
		FILTER_S = filter_s_t;
	}
	ap_uint<256> data;
	bool new_pair;
	bool last_pair;
	unsigned int FILTER_S;
};

typedef ap_uint<256> U1_Data0SIMDType;

struct U1_Data1PEChannelType{
	U1_Data1PEChannelType(){}
	U1_Data1PEChannelType(
			ap_uint<256> data_t
	){
		data = data_t;
	}
	U1_Data1PEChannelType(
			ap_uint<256> data_t,
			bool new_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		new_pair = new_pair_t;
		FILTER_S = filter_s_t;
	}
	U1_Data1PEChannelType(
			ap_uint<256> data_t,
			bool new_pair_t,
			bool last_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		new_pair = new_pair_t;
		last_pair = last_pair_t;
		FILTER_S = filter_s_t;
	}
	ap_uint<256> data;
	bool new_pair;
	bool last_pair;
	unsigned int FILTER_S;
};

typedef ap_uint<256> U1_Data1SIMDType;

struct U1_Data2PEChannelType{
	U1_Data2PEChannelType(){}
	U1_Data2PEChannelType(
			U1_data_t2 data_t){
		data = data_t;
	}
	U1_data_t2 data;
};

void U1_DataFeed0Head_Shim(
		U1_bus_t0* cin,
		stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > &fifo_transfer_cin,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H,
		uint LAYER_IN_IMG_W,
		uint LAYER_OUT_IMG_H,
		uint LAYER_OUT_IMG_W,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_FILTER_S,
		uint LAYER_BATCH,
		uint LAYER_STRIDE,
		stream<U1_ConfigInst> &fifo_kernel_config_out
);

void U1_DataFeed0Head(
		stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > &fifo_transfer_in,
		stream<U1_Data0TransferChannelType> &fifo_transfer_out0,
		stream<U1_ConfigInst> &fifo_kernel_config_in,
		stream<U1_ConfigInst> &fifo_kernel_config_out,
		stream<uint> &fifo_config_out0,
		stream<uint> &fifo_config_out1
);

void U1_DataFeed1Head_Shim(
		U1_bus_t1* weight,
		stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > &fifo_transfer_weight,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H,
		uint LAYER_IN_IMG_W,
		uint LAYER_OUT_IMG_H,
		uint LAYER_OUT_IMG_W,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_FILTER_S,
		uint LAYER_BATCH,
		uint LAYER_STRIDE
);

void U1_DataFeed1Head(
		stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > &fifo_transfer_in,
		stream<U1_Data1TransferChannelType> &fifo_transfer_out0,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
);

void U1_DataCollect2Head_Shim(
		U1_bus_t2* cout,
		stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > &fifo_transfer_cout,
		stream<U1_ConfigInst> &fifo_kernel_config_in,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H,
		uint LAYER_IN_IMG_W,
		uint LAYER_OUT_IMG_H,
		uint LAYER_OUT_IMG_W,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE
);

void U1_DataCollect2Head(
		stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > &fifo_transfer_out,
		stream<U1_Data2TransferChannelType> &fifo_transfer_in0,
		stream<uint> &fifo_config_in
);

void U1_DataFeed0Engine0_wrapper(
		stream<U1_Data0TransferChannelType> &fifo_transfer_in,
		stream<U1_Data0TransferChannelType> &fifo_transfer_out,
		stream<U1_Data0PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out0,
		stream<uint> &fifo_config_out1
);

void U1_DataFeed0EngineLast(
		stream<U1_Data0TransferChannelType> &fifo_transfer_in,
		stream<U1_Data0PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out1
);

void U1_DataFeed1Engine0_wrapper(
		stream<U1_Data1TransferChannelType> &fifo_transfer_in,
		stream<U1_Data1TransferChannelType> &fifo_transfer_out,
		stream<U1_Data1PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out0
);

void U1_DataFeed1EngineLast(
		stream<U1_Data1TransferChannelType> &fifo_transfer_in,
		stream<U1_Data1PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in
);

void U1_DataCollect2Engine0_wrapper(
		stream<U1_Data2TransferChannelType> &fifo_transfer_in,
		stream<U1_Data2TransferChannelType> &fifo_transfer_out,
		stream<U1_Data2PEChannelType> &fifo_collect_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in0,
		stream<uint> &fifo_config_in1,
		stream<uint> &fifo_config_out
);

void U1_DataCollect2EngineLast(
		stream<U1_Data2TransferChannelType> &fifo_transfer_out,
		stream<U1_Data2PEChannelType> &fifo_collect_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in0,
		stream<uint> &fifo_config_out
);

void U1_kernel(
		U1_bus_t0* cin,
		U1_bus_t1* weight,
		U1_bus_t2* cout,
		bool init,
		unsigned int FILTER_S
);

template<typename To, typename From>
inline To Reinterpret(const From& val){
 return reinterpret_cast<const To&>(val);
}

template<class data_t, class bus_t, int WIDTH>
data_t data_select(
		bus_t bus_data,
		uint offset
){
	data_t ret;
	ret = Reinterpret<data_t>((ap_uint<WIDTH>)bus_data(WIDTH-1 + offset*WIDTH, offset*WIDTH));
	return ret;
}


void conv2d_systolic(
	stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > &fifo_cin,
	stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > &fifo_weight,
	stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > &fifo_cout,
	ap_int<32> n, ap_int<32> c, ap_int<32> h, ap_int<32> w, 
	ap_int<32> k, ap_int<32> k_n
);