`timescale 1ns / 10ps
`include "util.vh"
//////////////////////////////////////////////////////////////////////////////////
// Design Name: vanilla
// Module Name: adder_tree
// Project Name: RBM_FPGA
// Target Devices: Xilinx Virtex Ultrascale
// Tool Versions: 
// Description: Inputing a vector of values to add, optimized addition via
//				binary adder tree.  Assume power of 2 number of nodes.
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: 
// 
//////////////////////////////////////////////////////////////////////////////////
module adder_tree # (

	//Number of precision bits for fixed point calculations
  	parameter PRECISION_BITS = 8,
  	// Number of nodes
  	parameter NUM_NODES = 4
  	// Machine clock frequency in Hz.

)(
  	input [PRECISION_BITS*NUM_NODES - 1:0] addends , //fixed point values to add
  	output reg [PRECISION_BITS - 1:0] result //Addition output
);
    integer i;
    reg [PRECISION_BITS - 1:0] temp_data;
    always @* begin
         temp_data = 0;
        for (i = 0; i < NUM_NODES; i = i + 1) 
            temp_data = temp_data + addends[i*PRECISION_BITS +: PRECISION_BITS];
        result = temp_data;
    end

    
endmodule