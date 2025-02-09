// MODELS
class ReceiptItem {
  description string
  quantity int
  quantity_unit string? @description(#"
    Unit of measurement for the quantity.
  "#)
  price float
}

class Receipt {
  line_items ReceiptItem[]
  total_amount float
  date string @description(#"
    Date of the transaction.
  "#)
  time string @description(#"
    Time of the transaction in 24-hour format.
  "#)

  business string @description(#"
    Name of the vendor or business that provided the receipt.
  "#)
  currency_code string @description(#"
    ISO 4217 Currency code for the currency used in the transaction. Infer from address if no currency symbol, or common symbol used among different countries.
  "#)
  address string?
  payment_method string? @description(#"
    Method of payment used for the transaction.
  "#)
}

// Original image processing function using the 90b model
function ExtractReceiptFromImage(img: image) -> Receipt {
    client LlamaVisionGroqClient
    prompt #"
        {{_.role("user")}}
        
        Extract details from this image of a receipt: {{ img }}

        The currency type may need to be inferred from the address info extracted. Example: 12345 Sunshine St, Sunnyville, CA 12345 is in the US, so return USD.

        {{ ctx.output_format }}
    "#
} 

// New image processing function using the 11b model
function ExtractReceiptFromImage11b(img: image) -> Receipt {
    client LlamaVisionGroqClient11b
    prompt #"
        {{_.role("user")}}
        
        Extract details from this image of a receipt: {{ img }}

        The currency type may need to be inferred from the address info extracted. Example: 12345 Sunshine St, Sunnyville, CA 12345 is in the US, so return USD.

        {{ ctx.output_format }}
    "#
} 

test TestExtraction {
  functions [ExtractReceiptFromImage]
  args {
    img { file "../test.jpg" }
  }
}