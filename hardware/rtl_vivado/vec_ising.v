`timescale 1ns / 1ps
`include "util.vh"

//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: vec_ising
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////



module vec_ising # (
  	// Machine clock frequency in Hz.
  	parameter RBM_CLOCK_FREQ = 50_000_000,
  	//Number of precision bits for fixed point calculations
  	parameter PRECISION_BITS = 4,
  	// Number of extra bits during calculation to prevent overflow
  	parameter OVERFLOW_BITS = 4,
    // Number of bits below fixed point
  	parameter P_INDEX = 3,
  	// What is the size of the address space of the URAM?
  	parameter URAM_AWIDTH = 12,
    // Number of nodes
    parameter NUM_NODES = 4,
    parameter NUM_NODES_BIT = 2,
    // Number of Colors 
    parameter NUM_COLORS = 4,
    parameter NUM_COLORS_BITS = 2,
  	//How many additions to do in a single cycle
  	parameter TREE_WIDTH = 8,
  	//Decimal and point location for sigmoid input
  	parameter N_IN = 5,
  	parameter P_IN = 3,
    //Decimal and point location for sigmoid output
    parameter N_OUT = 16,
    parameter P_OUT = N_OUT - 1, 
    parameter RNG_LEN = 32

) (
	input clk,			//Machine clock
	input rst,
	input rst_state,
    //Pipelined weight values, operating on each row of weight matrix
    input[(PRECISION_BITS*NUM_NODES) - 1:0] weights,

    input weights_valid, //indicates whether input weights are valid

    output [URAM_AWIDTH - 1:0] mem_addr,
	//Used to clamp output to either 0 or 1
	output [NUM_NODES*NUM_COLORS_BITS - 1:0]  node_data,	//Output data to computer
    output reg state_data_valid, /// Once all the spins are sampled, this will be on and data will be stored in the memory
    output reg next_node_pointer ////Pointer to move on next address it will be one only when all the colors are calculated. 
);


    reg [URAM_AWIDTH-1:0] state_counter;
    reg [`log2(NUM_COLORS_BITS) - 1:0] col_counter;
    reg [NUM_COLORS_BITS*NUM_NODES - 1:0] global_counter; /// Actually calculate the states among the color and node 

    reg [NUM_NODES*NUM_COLORS_BITS - 1:0] state_nodes;

    wire [PRECISION_BITS + OVERFLOW_BITS - 1:0] delta_energy;
    wire node_update_val; 
    
    node_update # (
        .PRECISION_BITS(PRECISION_BITS),
        .P_INDEX(P_INDEX),
        .OVERFLOW_BITS(OVERFLOW_BITS),
        .NUM_NODES(NUM_NODES),
        .NUM_NODES_BIT(NUM_NODES_BIT),
        .NUM_COLORS(NUM_COLORS),
        .NUM_COLORS_BITS(NUM_COLORS_BITS),
        .RBM_CLOCK_FREQ(RBM_CLOCK_FREQ),
        .N_IN(N_IN),
        .P_IN(P_IN),
        .N_OUT(N_OUT),
        .P_OUT(P_OUT),
        .RNG_LEN(RNG_LEN)
    ) sample_node(
        .clk(clk),  
        .rst(rst),
        .rst_state(rst_state),
        // input[PRECISION_BITS - 1:0] bias, // Not using bias if needed weight ii can be used or diagonal of an array
        .nodes(state_nodes),   
        .weights(weights),  //Fixed point weights
        .node_count(state_counter),
        .color_bit_count(col_counter),
        .sigmoid_input(delta_energy), //For use with hitting time engine May be for energy calculation later for parallel tempering implementation
        .new_val(node_update_val) 	//Binary result, new node value
    );

    

    initial begin 
        state_counter <= 'b0;  
        next_node_pointer <= 'b1; 
    end

    always @(posedge clk) begin
        if(rst) begin
            state_nodes <= 'b0;
            state_counter <= 'b0;
            col_counter <= 'b0;
            global_counter <= 'b0;
            next_node_pointer <= 'b0; 
        end else begin
//            if(weights_valid) begin
//                    if (rst_state) state_nodes <= {(NUM_NODES*NUM_COLORS_BITS){1'b0}};
//                    else state_nodes[global_counter] <= node_update_val; 
//            end

            if(~ ((state_counter == NUM_NODES-1) && (col_counter ==NUM_COLORS_BITS-1)))begin
                state_data_valid <= 'b0;
                if(weights_valid) begin
                    if (rst_state) state_nodes <= {(NUM_NODES*NUM_COLORS_BITS){1'b0}};
                    else state_nodes[global_counter] <= node_update_val; 
                    if (col_counter == NUM_COLORS_BITS-1) col_counter <= 'b0;
                    else col_counter <= col_counter+1;
                    global_counter <= global_counter+1;
                    if(col_counter ==NUM_COLORS_BITS-1) state_counter <= state_counter+1;
        
                    if(col_counter ==NUM_COLORS_BITS-2) begin // -2 because a clock cycle gap
                        // state_counter <= state_counter+1;
                        next_node_pointer <= 1'b1;
                    end
                    else begin
                        next_node_pointer <= 1'b0;
                    end
                end 
            // else if (col_counter == NUM_COLORS_BITS-2) begin 
            //     col_counter <= col_counter+1;
            //     next_node_pointer <= 1'b1;
            //     state_nodes[global_counter] <= node_update_val; 
            //     global_counter <= global_counter+1;

            // end
            end else begin
                next_node_pointer <= 1'b0;
                state_data_valid <= 'b1;
                state_counter <= 'b0;
                col_counter <= 'b0;
                global_counter <= 'b0;
            end
        end
    end
    assign node_data = state_nodes;
    assign mem_addr = state_counter;
endmodule