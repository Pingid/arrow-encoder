# Arrow Encoder

A Rust library providing flexible abstractions for encoding row-oriented data into Apache Arrow `RecordBatch`es.

The core goal is to simplify converting data from various sources (like JSON) into the columnar Arrow format efficiently.

---

## Features

- **Generic Encoding Framework**: Core traits (`RowBuilder`, `BuilderFromField`) allow for implementing custom encoders for any data source.
- **Automatic Batching**: The `FieldBatchEncoder` handles the logic of collecting rows and flushing them into `RecordBatch`es of a specified size.
- **Schema-Driven**: Encoders are constructed from an Arrow `Schema`, ensuring type safety and correctness.
- **Iterator-Based**: Processes data streams efficiently using Rust's `Iterator` trait.
- **Ready-to-use Encoders**: Ships with a `serde_json::Value` encoder out of the box.

---

## Core Abstractions

This library is built around a few key traits and structs:

- `RowBuilder`: A trait for implementing the logic of appending a single typed row value to a specific Arrow `ArrayBuilder`.
- `RowBatcher`: Manages a collection of column builders (`RowBuilder`s) and orchestrates the creation of a `RecordBatch` from rows.
- `FieldBatchEncoder`: An iterator that wraps a `RowBatcher` to consume any row-based data iterator and produce an iterator of `Result<RecordBatch, E>`.

---

## Usage Example

Here's how to encode an iterator of `serde_json::Value` objects into an Arrow `RecordBatch`.

First, add this to your `Cargo.toml`:

```toml
[dependencies]
arrow-encoder = "0.0.1"
arrow = "56"
serde_json = "1.0"
```

```rust
use std::sync::Arc;
use arrow::datatypes::{DataType, Field, Schema};
use serde_json::json;
use arrow_encoder::JsonEncoder; // The concrete encoder for serde_json::Value

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define the target Arrow schema.
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, true),
        Field::new("is_student", DataType::Boolean, true),
    ]));

    // 2. Prepare your source data (any iterator over serde_json::Value).
    let values = vec![
        json!({"name": "Alice", "age": 30, "is_student": false}),
        json!({"name": "Bob", "age": 25, "is_student": true}),
        json!({"name": "Charlie", "age": 42, "is_student": null}),
        json!({"name": "David", "age": null}),
    ];

    // 3. Create a JsonEncoder from the schema, data iterator, and a batch size.
    let mut encoder = JsonEncoder::from_schema(schema, values.into_iter(), 1024)?;

    // 4. The encoder is an iterator that yields RecordBatches.
    //    Here, we expect only one batch since our data is small.
    if let Some(Ok(batch)) = encoder.next() {
        // The arrow pretty_print feature is useful for visualization.
        batch.pretty_print().unwrap();
    }

    Ok(())
}
```

---

## Available Encoders

- ✅ **JSON**: `JsonEncoder` for `serde_json::Value`.
- 🔜 **More coming soon!** The framework is designed to be easily extensible.

---
