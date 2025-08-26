`timescale 1ns / 1ps
`include "rng.vh"
`include "util.vh"

//////////////////////////////////////////////////////////////////////////////////
// Design Name: vanilla
// Module Name: node_update
// Project Name: RBM_FPGA
// Target Devices: Xilinx Virtex Ultrascale
// Tool Versions: 
// Description: Updates node values given previous node, weights, and biases
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: 
// 
//////////////////////////////////////////////////////////////////////////////////
module node_update # (
  	//Number of precision bits for fixed point calculations
  	parameter PRECISION_BITS = 4,
  	// Number of bits below fixed point
  	parameter P_INDEX = 3,
  	//Extra bits to prevent overflow in additions
  	parameter OVERFLOW_BITS = 4,
    // Number of nodes
    parameter NUM_NODES = 4,
    parameter NUM_NODES_BIT = 2,
    // Number of Colors 

    parameter NUM_COLORS = 4,
    parameter NUM_COLORS_BITS = 2,
  	// Machine clock frequency in Hz.
  	parameter RBM_CLOCK_FREQ = 50_000_000,
  	//Decimal and point location for sigmoid input
  	parameter N_IN = 5,
  	parameter P_IN = 3,
  	//Decimal and point location for sigmoid output
	parameter N_OUT = 16,
	parameter P_OUT = 15,
	parameter RNG_LEN = 32 

) (
  	input clk,      //Machine clock
	input rst,
	input rst_state,
  	// input[PRECISION_BITS - 1:0] bias,
  	input[NUM_NODES*NUM_COLORS_BITS - 1:0] nodes,   //Previous node values
	input[PRECISION_BITS*NUM_NODES - 1:0] weights,  //Fixed point weights
    input [NUM_NODES_BIT-1:0] node_count,
    input [`log2(NUM_COLORS_BITS)-1:0] color_bit_count,
	output reg [PRECISION_BITS + OVERFLOW_BITS - 1:0] sigmoid_input, //For use with hitting time engine
	output new_val 	//Binary result, new node value
);


	//Perform vector multiplication of weight row and input nodes
	wire [PRECISION_BITS + OVERFLOW_BITS - 1:0] vecmul_out;
	vecmul_gc  #(
		.PRECISION_BITS(PRECISION_BITS),
		.OVERFLOW_BITS(OVERFLOW_BITS),
		.NUM_NODES(NUM_NODES),
        .NUM_NODES_BIT(NUM_NODES_BIT),
        .NUM_COLORS(NUM_COLORS),
        .NUM_COLORS_BITS(NUM_COLORS_BITS)
	) vec_multiplier (
  		.nodes(nodes),
		.weights(weights),
        .node_count(node_count),
        .color_bit_count(color_bit_count),
 		.product(vecmul_out) 
	);

	//Doing this ensures nothing overflows, as we keep everything in the higher bit count domain
	// wire [PRECISION_BITS + OVERFLOW_BITS - 1:0] bias_extended = $signed(bias);
	always @(*) begin
	   sigmoid_input = vecmul_out;// + bias_extended;<<3
	end

	//Pass through non linear squishing function
	//sigmoid function now operates with more bits
	wire [N_OUT - 1:0] sigmoid_output;
	sigmoid_func #(
	.N_IN(N_IN),
	.P_IN(P_IN),
	.N_OUT(N_OUT),
	.P_OUT(P_OUT),
	.P_INDEX(P_INDEX),
	.PRECISION_BITS(PRECISION_BITS+OVERFLOW_BITS)
	) nonlinear_machine (
		.x(sigmoid_input),
		.y(sigmoid_output)
	);


	//array of random numbers
	//To take advantage of pipelining, we are using a single 
	// hardworking RNG to generate random numbers on a circular counter
	// reg [P_OUT-1:0] rand_array[NUM_NODES-1:0];
	// reg [`log2(NUM_NODES)-1:0] rand_counter;
	wire [P_OUT - 1:0] rand;
	lfsr_lookahead #(
			.N(P_OUT),
			.LENGTH(RNG_LEN),
			.TAPS(`RNG_TAPS(RNG_LEN)),
			.SEED(2)
		) rng (
			.clk(clk), .rst(rst_state),
			.out(rand)
		);
	// always @(posedge clk) begin
	// 	if(rst) begin
	// 		rand_counter <= 'b0;
	// 	end else begin
	// 		if (rand_counter >= NUM_NODES) begin
	// 			rand_counter <= 'b0; 
	// 		end else begin
	// 			rand_counter <= rand_counter + 1;
	// 		end
	// 		rand_array[rand_counter] <= rand; 
	// 	end
	// end

    //Comparing against a random number
	wire [N_OUT - 1:0] random_num;
	assign random_num = {{(N_OUT-P_OUT){1'b0}}, rand};

	assign  new_val = sigmoid_output > random_num;

endmodule