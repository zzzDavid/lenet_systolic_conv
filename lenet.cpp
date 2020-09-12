#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <stdint.h>
#include "conv.h"

void post_process(float dense1[][10], float lenet[][10]) {
  float compute0[1000];
  for (ap_int<32> i7 = 0; i7 < 1000; ++i7) {
    ap_int<32> max;
    max = 0;
    for (ap_int<32> ra6 = 0; ra6 < 10; ++ra6) {
      max = ((ap_int<32>)hls::max(dense1[i7][ra6], ((float)max)));
    }
    compute0[i7] = ((float)max);
  }
  float compute1[1000];
  for (ap_int<32> i8 = 0; i8 < 1000; ++i8) {
    ap_int<32> sum2;
    sum2 = 0;
    for (ap_int<32> ra7 = 0; ra7 < 10; ++ra7) {
      sum2 = ((ap_int<32>)(sqrt(((float)(dense1[i8][ra7] - compute0[i8]))) + ((float)sum2)));
    }
    compute1[i8] = ((float)sum2);
  }
  float update0;
  for (ap_int<32> i9 = 0; i9 < 1000; ++i9) {
    for (ap_int<32> j3 = 0; j3 < 10; ++j3) {
      lenet[i9][j3] = ((float)(sqrt(((float)(dense1[i9][j3] - compute0[i9]))) / ((float)compute1[i9])));
    }
  }
}


void default_function(float input_image[1000][1][28][28], float weight_conv1[20][1][5][5], float weight_conv2[50][20][5][5], float weight_fc1[500][800], float weight_fc2[10][500], float lenet[1000][10]) {
  float _top;
  float pad_temp[1000][1][28][28];
  for (ap_int<32> indices = 0; indices < 1000; ++indices) {
    for (ap_int<32> index_tuple = 0; index_tuple < 28; ++index_tuple) {
      for (ap_int<32> i = 0; i < 28; ++i) {
        pad_temp[indices][0][index_tuple][i] = input_image[indices][0][index_tuple][i];
      }
    }
  }
  float conv2d[1000][20][24][24];
  // for (ap_int<32> nn = 0; nn < 1000; ++nn) {
  //   for (ap_int<32> ff = 0; ff < 20; ++ff) {
  //     for (ap_int<32> yy = 0; yy < 24; ++yy) {
  //       for (ap_int<32> xx = 0; xx < 24; ++xx) {
  //         float sum;
  //         sum = 0.000000e+00f;
  //         for (ap_int<32> ry = 0; ry < 5; ++ry) {
  //           for (ap_int<32> rx = 0; rx < 5; ++rx) {
  //             sum = ((pad_temp[nn][0][(yy + ry)][(xx + rx)] * weight_conv1[ff][0][ry][rx]) + sum);
  //           }
  //         }
  //         conv2d[nn][ff][yy][xx] = sum;
  //       }
  //     }
  //   }
  // }
  #pragma HLS DATAFLOW
    stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > fifo_cin;
    stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > fifo_weight;
    stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > fifo_cout;
    #pragma HLS STREAM variable=fifo_cin depth=4 
    #pragma HLS STREAM variable=fifo_weight depth=4
    #pragma HLS STREAM variable=fifo_cout depth=4

	for (ap_int<32> i1 = 0; i1 < 1000; i1++){
		for (ap_int<32> i2 = 0; i2 < 1; i2++){
			for (ap_int<32> i3 = 0; i3 < 28; i3++) {
				for (ap_int<32> i4 = 0; i4 < 28; i4++){
					fifo_cin.write(pad_temp[i1][i2][i3][i4]);
				}
			}
		}
	}
	for (ap_int<32> i1 = 0; i1 < 20; i1++){
		for (ap_int<32> i2 = 0; i2 < 1; i2++){
			for (ap_int<32> i3 = 0; i3 < 5; i3++) {
				for (ap_int<32> i4 = 0; i4 < 5; i4++){
					fifo_weight.write(weight_conv1[i1][i2][i3][i4]);
				}
			}
		}
	}
  conv2d_systolic(fifo_cin, fifo_weight, fifo_cout, 1000, 1, 28, 28, 5, 20);
  // read output
	for (ap_int<32> i1 = 0; i1 < 100; i1++){
		for (ap_int<32> i2 = 0; i2 < 20; i2++){
			for (ap_int<32> i3 = 0; i3 < 24; i3++) {
				for (ap_int<32> i4 = 0; i4 < 24; i4 += U1_DATA0_FC_SIMD_FACTOR){
					for (ap_int<32> i5 = 0; i5 < U1_DATA0_FC_SIMD_FACTOR; i5++){
						conv2d[i1][i2][i3][i4+i5] = fifo_cout.read();
					}
				}
			}
		}
	}

  float tanh1[1000][20][24][24];
  for (ap_int<32> args = 0; args < 1000; ++args) {
    for (ap_int<32> args0 = 0; args0 < 20; ++args0) {
      for (ap_int<32> args1 = 0; args1 < 24; ++args1) {
        for (ap_int<32> args2 = 0; args2 < 24; ++args2) {
          tanh1[args][args0][args1][args2] = (float)sqrt((float)conv2d[args][args0][args1][args2]);
        }
      }
    }
  }
  float max_pool2d[1000][20][12][12];
  for (ap_int<32> i1 = 0; i1 < 1000; ++i1) {
    for (ap_int<32> c = 0; c < 20; ++c) {
      for (ap_int<32> h = 0; h < 12; ++h) {
        for (ap_int<32> w = 0; w < 12; ++w) {
          float reducer0;
          reducer0 = -1.000000e+00f;
          for (ap_int<32> ra0 = 0; ra0 < 2; ++ra0) {
            for (ap_int<32> ra1 = 0; ra1 < 2; ++ra1) {
              reducer0 = hls::max(tanh1[i1][c][((h * 2) + ra0)][((w * 2) + ra1)], reducer0);
            }
          }
          max_pool2d[i1][c][h][w] = reducer0;
        }
      }
    }
  }
  float pad_temp1[1000][20][12][12];
  for (ap_int<32> indices1 = 0; indices1 < 1000; ++indices1) {
    for (ap_int<32> not_zero = 0; not_zero < 20; ++not_zero) {
      for (ap_int<32> index_tuple1 = 0; index_tuple1 < 12; ++index_tuple1) {
        for (ap_int<32> i2 = 0; i2 < 12; ++i2) {
          pad_temp1[indices1][not_zero][index_tuple1][i2] = max_pool2d[indices1][not_zero][index_tuple1][i2];
        }
      }
    }
  }
  float conv2d1[1000][50][8][8];
  for (ap_int<32> nn1 = 0; nn1 < 1000; ++nn1) {
    for (ap_int<32> ff1 = 0; ff1 < 50; ++ff1) {
      for (ap_int<32> yy1 = 0; yy1 < 8; ++yy1) {
        for (ap_int<32> xx1 = 0; xx1 < 8; ++xx1) {
          float sum1;
          sum1 = 0.000000e+00f;
          for (ap_int<32> rc = 0; rc < 20; ++rc) {
            for (ap_int<32> ry1 = 0; ry1 < 5; ++ry1) {
              for (ap_int<32> rx1 = 0; rx1 < 5; ++rx1) {
                sum1 = ((pad_temp1[nn1][rc][(yy1 + ry1)][(xx1 + rx1)] * weight_conv2[ff1][rc][ry1][rx1]) + sum1);
              }
            }
          }
          conv2d1[nn1][ff1][yy1][xx1] = sum1;
        }
      }
    }
  }
  float tanh2[1000][50][8][8];
  for (ap_int<32> args3 = 0; args3 < 1000; ++args3) {
    for (ap_int<32> args01 = 0; args01 < 50; ++args01) {
      for (ap_int<32> args11 = 0; args11 < 8; ++args11) {
        for (ap_int<32> args21 = 0; args21 < 8; ++args21) {
          tanh2[args3][args01][args11][args21] = ((float)sqrt(((float)conv2d1[args3][args01][args11][args21])));
        }
      }
    }
  }
  float max_pool2d1[1000][50][4][4];
  for (ap_int<32> i3 = 0; i3 < 1000; ++i3) {
    for (ap_int<32> c1 = 0; c1 < 50; ++c1) {
      for (ap_int<32> h1 = 0; h1 < 4; ++h1) {
        for (ap_int<32> w1 = 0; w1 < 4; ++w1) {
          float reducer1;
          reducer1 = -1.000000e+00f;
          for (ap_int<32> ra2 = 0; ra2 < 2; ++ra2) {
            for (ap_int<32> ra3 = 0; ra3 < 2; ++ra3) {
              reducer1 = hls::max(tanh2[i3][c1][((h1 * 2) + ra2)][((w1 * 2) + ra3)], reducer1);
            }
          }
          max_pool2d1[i3][c1][h1][w1] = reducer1;
        }
      }
    }
  }
  float flatten[1000][800];
  for (ap_int<32> i4 = 0; i4 < 1000; ++i4) {
    for (ap_int<32> j = 0; j < 800; ++j) {
      flatten[i4][j] = max_pool2d1[i4][(j / 16)][((j / 4) % 4)][(j % 4)];
    }
  }
  float dense[1000][500];
  for (ap_int<32> i5 = 0; i5 < 1000; ++i5) {
    for (ap_int<32> j1 = 0; j1 < 500; ++j1) {
      float reducer2;
      reducer2 = 0.000000e+00f;
      for (ap_int<32> ra4 = 0; ra4 < 800; ++ra4) {
        reducer2 = ((flatten[i5][ra4] * weight_fc1[j1][ra4]) + reducer2);
      }
      dense[i5][j1] = reducer2;
    }
  }
  float tanh3[1000][500];
  for (ap_int<32> args4 = 0; args4 < 1000; ++args4) {
    for (ap_int<32> args02 = 0; args02 < 500; ++args02) {
      tanh3[args4][args02] = ((float)sqrt(((float)dense[args4][args02])));
    }
  }
  float dense1[1000][10];
  for (ap_int<32> i6 = 0; i6 < 1000; ++i6) {
    for (ap_int<32> j2 = 0; j2 < 10; ++j2) {
      float reducer3;
      reducer3 = 0.000000e+00f;
      for (ap_int<32> ra5 = 0; ra5 < 500; ++ra5) {
        reducer3 = ((tanh3[i6][ra5] * weight_fc2[j2][ra5]) + reducer3);
      }
      dense1[i6][j2] = reducer3;
    }
  }
  

  post_process(dense1, lenet);
}

