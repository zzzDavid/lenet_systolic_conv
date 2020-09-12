open_project lenet
set_top default_function
add_files ./conv.h
add_files ./conv.cpp
add_files ./lenet.cpp
open_solution "lenet_systolic"
set_part {xc7vx690tffg1761-2} -tool vivado
create_clock -period 3 -name default
config_interface -m_axi_addr64 -m_axi_offset off -register_io off
# csim_design -compiler gcc
csynth_design
