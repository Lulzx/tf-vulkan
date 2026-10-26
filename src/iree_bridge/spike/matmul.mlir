module {
  func.func @matmul(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %empty = tensor.empty() : tensor<4x4xf32>
    %zero = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
                        outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %mm : tensor<4x4xf32>
  }
}
