`timescale 1ns / 10ps
`include "util.vh"

//////////////////////////////////////////////////////////////////////////////////
// Create Date: 1/09/2019 1:22:35 PM
// Design Name: vanilla
// Module Name: vecmul
// Project Name: RBM_FPGA
// Target Devices: Xilinx Virtex Ultrascale
// Tool Versions:
// Description: An implementation of "vector multiplication" by masking values
//		          then performing addition on the survivors
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////
module vecmul_gc # (

  //Number of precision bits for fixed point calculations
  parameter PRECISION_BITS = 4,
  // Number of nodes
  parameter NUM_NODES = 4,
  parameter NUM_NODES_BIT = 2,
  // Number of Colors 

  parameter NUM_COLORS = 4,
  parameter NUM_COLORS_BITS = 2,

  // Extra overflow bits
  parameter OVERFLOW_BITS = 4

) (
  // input clk,      //Machine clock
  input [NUM_NODES*NUM_COLORS_BITS - 1:0] nodes,   //Binary node values
  input [PRECISION_BITS*NUM_NODES- 1:0] weights, //fixed point weight values
  input [NUM_NODES_BIT-1:0] node_count,
  input [`log2(NUM_COLORS_BITS)-1:0] color_bit_count,
  output reg [(PRECISION_BITS + OVERFLOW_BITS) - 1:0] product //Vector multiplication output
);

  localparam PAD_BITS = PRECISION_BITS + OVERFLOW_BITS;

  localparam MAX_POS = {1'b0, {(PRECISION_BITS-1){1'b1}}};
  localparam MAX_NEG = {1'b1, {(PRECISION_BITS-2){1'b0}}, 1'b1};

  wire [PAD_BITS*NUM_NODES - 1:0] masked_weights;



  localparam MUX_WEIGHT = 1 << (NUM_COLORS_BITS*2);

  wire [NUM_COLORS_BITS-1:0] curr_node_select_bits;
  reg [NUM_COLORS_BITS-1:0] curr_node_select_bits_1;
  reg [NUM_COLORS_BITS-1:0] curr_node_select_bits_0;

  assign curr_node_select_bits = nodes[(node_count+1)*NUM_COLORS_BITS-1 -: NUM_COLORS_BITS];

  wire [NUM_COLORS_BITS-1:0] operator_0 [0: NUM_COLORS_BITS-1];
  wire [NUM_COLORS_BITS-1:0] operator_1 [0: NUM_COLORS_BITS-1];

  genvar cn;
  generate 
   for(cn = 0; cn < NUM_COLORS_BITS; cn = cn+1) begin
      if (cn == 0) begin
          assign operator_0[cn] = {{(NUM_COLORS_BITS-1){1'b1}}, 1'b0}; // Change the 0th bit
          assign operator_1[cn] = {{(NUM_COLORS_BITS-1){1'b0}}, 1'b1};
      end
      else if (cn == NUM_COLORS_BITS-1) begin
          assign operator_0[cn] = {1'b0, {(NUM_COLORS_BITS-1){1'b1}}};              // Change the (NUM_COLORS_BITS-1)th bit (MSB)
          assign operator_1[cn] = {1'b1, {(NUM_COLORS_BITS-1){1'b0}}};
      end
      else begin
          assign operator_0[cn] = {{(NUM_COLORS_BITS-cn-1){1'b1}},1'b0, {(cn){1'b1}}};   // Remove the nth bit and concatenate bits around it
          assign operator_1[cn] = {{(NUM_COLORS_BITS-cn-1){1'b0}},1'b1, {(cn){1'b0}}}; 
      end 
   end
  endgenerate 

  always @(*) begin
      curr_node_select_bits_1 = curr_node_select_bits | operator_1[color_bit_count]; // Change the 0th bit
      curr_node_select_bits_0 = curr_node_select_bits & operator_0[color_bit_count];
  end



  ///// For testing visibility ////
//  wire [MUX_WEIGHT*PRECISION_BITS-1:0] mux_weight_in_test [0: NUM_NODES-1];
//  wire [PRECISION_BITS-1:0] mux_out_1_test [0: NUM_NODES-1];
//  wire [PRECISION_BITS-1:0] mux_out_0_test [0: NUM_NODES-1];
//  wire [NUM_COLORS_BITS*2 -1 : 0] select_0_test [0: NUM_NODES-1]; 
//  wire [NUM_COLORS_BITS*2 -1 : 0] select_1_test [0: NUM_NODES-1]; 

  genvar i;
  integer j, k;
  generate
    // Mask and pad weights
    for (i = 0; i < NUM_NODES; i = i + 1) begin
        reg [MUX_WEIGHT*PRECISION_BITS-1:0] mux_weight_in; 
        reg [PRECISION_BITS-1:0] mux_out_1; 
        reg [PRECISION_BITS-1:0] mux_out_0;
          // wire [PRECISION_BITS - 1:0] weight = weights[PRECISION_BITS*i +: PRECISION_BITS]; /// Weight of connection i, j
        always @(*) begin

          for (j =0; j<(1 << NUM_COLORS_BITS); j = j+1 ) begin 
            for (k =0; k<(1 << NUM_COLORS_BITS); k = k+1 ) begin 
              if ((j >= NUM_COLORS) || (k >= NUM_COLORS)) begin 
                mux_weight_in[((k+(1<<NUM_COLORS_BITS)*j))*PRECISION_BITS +: PRECISION_BITS] = weights[(PRECISION_BITS * i) +: PRECISION_BITS]; // weight;
              end
              else if (j == k) begin
                mux_weight_in[(k+(1<<NUM_COLORS_BITS)*j)*PRECISION_BITS +: PRECISION_BITS] = weights[PRECISION_BITS*i +: PRECISION_BITS];
              end
              else begin 
                mux_weight_in[(k+(1<<NUM_COLORS_BITS)*j)*PRECISION_BITS +: PRECISION_BITS] = {(PRECISION_BITS){1'b0}};
              end 
            end 
          mux_out_1 =  mux_weight_in[({curr_node_select_bits_1, nodes[(i+1)*NUM_COLORS_BITS-1 -: NUM_COLORS_BITS]}+1)*(PRECISION_BITS)-1 -:(PRECISION_BITS)];
          mux_out_0 =  mux_weight_in[({curr_node_select_bits_0, nodes[(i+1)*NUM_COLORS_BITS-1 -: NUM_COLORS_BITS]}+1)*(PRECISION_BITS)-1 -:(PRECISION_BITS)];

          end

        end
//          assign mux_weight_in_test[i] =  mux_weight_in;
//          assign mux_out_1_test[i] =  mux_out_1;
//          assign mux_out_0_test[i] =  mux_out_0;
          // assign select_0_test[i] = 
          // /////// Call the Mux Operation //////
          // vecmul_mux
          //   #(.SELECT_BITS(NUM_COLORS_BITS*2),
          //     .NUM_INPUTS(1 << (NUM_COLORS_BITS*2)),
          //     .INPUT_SIZE(PRECISION_BITS)
          //     ) mux_1 (
          //       .select({curr_node_select_bits_1, nodes[(i+1)*NUM_COLORS_BITS-1 -: NUM_COLORS_BITS]}),
          //       .in_arr(mux_weight_in),
          //       .out(mux_out_1) 
          //   );

          // vecmul_mux
          //   #(.SELECT_BITS(NUM_COLORS_BITS*2),
          //     .NUM_INPUTS(1 << (NUM_COLORS_BITS*2)),
          //     .INPUT_SIZE(PRECISION_BITS)
          //     ) mux_0 (
          //       .select({curr_node_select_bits_0, nodes[(i+1)*NUM_COLORS_BITS-1 -: NUM_COLORS_BITS]}),
          //       .in_arr(mux_weight_in),
          //       .out(mux_out_0) 
          //   );
          wire [PAD_BITS - 1:0] pad_weight = -$signed(mux_out_1)+$signed(mux_out_0); // - dE/ds goes into activation function
          assign masked_weights[PAD_BITS*(i + 1) - 1: PAD_BITS*i] = pad_weight;  
    end
  endgenerate

  //Instantiate adder implementation for efficient computation
  //Forcing proper truncation for signed numbers
  wire [PAD_BITS - 1:0] pad_out;
  adder_tree  #(
    //Number of precision bits for fixed point calculations
    .PRECISION_BITS(PAD_BITS),
    // Number of nodes
    .NUM_NODES(NUM_NODES)
  ) tree (
    .addends(masked_weights),
    .result(pad_out)
  );
  
  //Registering the output of the adder tree (the tree has a lot of latency)

  // reg [PAD_BITS - 1:0] pad_reg;
  // always @(posedge clk) begin
  //   pad_reg <= pad_out;
  // end
  
  always @* begin
    product = pad_out;
  end



  

endmodule



