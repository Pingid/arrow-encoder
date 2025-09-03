use std::any::Any;
use std::sync::Arc;

use arrow::{
    array::{ArrayRef, RecordBatch},
    datatypes::{Field, Schema},
    error::ArrowError,
};

mod json;
pub use json::*;

pub trait RowBuilder {
    type Error;
    type Row;

    fn append_row(&mut self, field: &Arc<Field>, value: &Self::Row) -> Result<(), Self::Error>;

    fn finish(&mut self) -> ArrayRef;
}

pub trait BuilderFromField {
    type Error;
    fn try_from_field(field: &Arc<Field>) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

pub struct RowBatcher<C> {
    schema: Arc<Schema>,
    cols: Vec<C>,
    batch_size: usize,
    rows: usize,
}

impl<C: RowBuilder<Error = E> + BuilderFromField<Error = E>, E> RowBatcher<C> {
    pub fn from_schema(schema: Arc<Schema>, batch_size: usize) -> Result<Self, E> {
        let cols = schema
            .fields()
            .iter()
            .map(|f| C::try_from_field(f))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(schema, cols, batch_size)
    }
}

impl<C: RowBuilder> RowBatcher<C> {
    pub fn new(schema: Arc<Schema>, cols: Vec<C>, batch_size: usize) -> Result<Self, C::Error> {
        Ok(Self {
            schema,
            cols,
            batch_size,
            rows: 0,
        })
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.rows >= self.batch_size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn push_row(&mut self, row: &C::Row) -> Result<(), C::Error> {
        // For each column, fetch value from the row via FieldDef and append
        for (col, f) in self.cols.iter_mut().zip(self.schema.fields().iter()) {
            col.append_row(f, row)?;
        }
        self.rows += 1;
        Ok(())
    }

    pub fn finish_batch(&mut self) -> Result<RecordBatch, ArrowError> {
        let arrays: Vec<ArrayRef> = self.cols.iter_mut().map(|c| c.finish()).collect();
        self.rows = 0;
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

pub struct FieldBatchEncoder<C: RowBuilder, V: Iterator<Item = C::Row>> {
    values: V,
    batcher: RowBatcher<C>,
}

impl<
    C: RowBuilder<Error = E> + BuilderFromField<Error = E>,
    V: Iterator<Item = C::Row>,
    E: From<ArrowError>,
> FieldBatchEncoder<C, V>
{
    pub fn from_schema(schema: Arc<Schema>, values: V, batch_size: usize) -> Result<Self, E> {
        let batcher = RowBatcher::<C>::from_schema(schema, batch_size)?;
        Ok(Self { values, batcher })
    }
}

impl<E: From<ArrowError>, C: RowBuilder<Error = E>, V: Iterator<Item = C::Row>>
    FieldBatchEncoder<C, V>
{
    pub fn new(
        schema: Arc<Schema>,
        cols: Vec<C>,
        values: V,
        batch_size: usize,
    ) -> Result<Self, C::Error> {
        let batcher = RowBatcher::new(schema, cols, batch_size)?;
        Ok(Self { values, batcher })
    }

    /// Fills up to `batch_size` rows or until the input iterator is exhausted.
    pub fn write_next(&mut self) -> Result<Option<RecordBatch>, C::Error> {
        while !self.batcher.is_full() {
            match self.values.next() {
                Some(row) => self.batcher.push_row(&row)?,
                None => break,
            }
        }

        if self.batcher.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.batcher.finish_batch()?))
        }
    }
}

impl<E: From<ArrowError>, C: RowBuilder<Error = E>, V: Iterator<Item = C::Row>> Iterator
    for FieldBatchEncoder<C, V>
{
    type Item = Result<RecordBatch, E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.write_next().transpose()
    }
}

macro_rules! make_encoder {
    ($name:ident { $(($variant:ident, $builder:ty)),* $(,)? }) => {
        pub enum $name {
            $( $variant($builder) ),*
        }

        impl $name {
            pub fn append_null(&mut self) {
                match self {
                    $( $name::$variant(builder) => builder.append_null(), )*
                }
            }
        }

        impl ArrayBuilder for $name {
            fn as_any(&self) -> &dyn Any {
                match self { $( $name::$variant(b) => b.as_any(), )* }
            }
            fn as_any_mut(&mut self) -> &mut dyn Any {
                match self { $( $name::$variant(b) => b.as_any_mut(), )* }
            }
            fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
                match *self { $( $name::$variant(b) => Box::new(b).into_box_any(), )* }
            }
            fn len(&self) -> usize {
                match self { $( $name::$variant(b) => b.len(), )* }
            }
            fn finish(&mut self) -> ArrayRef {
                match self { $( $name::$variant(b) => Arc::new(b.finish()), )* }
            }
            fn finish_cloned(&self) -> ArrayRef {
                match self { $( $name::$variant(b) => Arc::new(b.finish_cloned()), )* }
            }
        }

        impl ArrayBuilder for Box<$name> {
            fn as_any(&self) -> &dyn Any {
                match self.as_ref() { $( $name::$variant(b) => b.as_any(), )* }
            }
            fn as_any_mut(&mut self) -> &mut dyn Any {
                match self.as_mut() { $( $name::$variant(b) => b.as_any_mut(), )* }
            }
            fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
                unimplemented!()
            }
            fn len(&self) -> usize {
                match self.as_ref() { $( $name::$variant(b) => b.len(), )* }
            }
            fn finish(&mut self) -> ArrayRef {
                match self.as_mut() { $( $name::$variant(b) => Arc::new(b.finish()), )* }
            }
            fn finish_cloned(&self) -> ArrayRef {
                match self.as_ref() { $( $name::$variant(b) => Arc::new(b.finish_cloned()), )* }
            }
        }
    };
}

use arrow::array::MapFieldNames;
use arrow::array::StructArray;
use arrow::array::*;
use arrow::datatypes::IntervalUnit::*;
use arrow::datatypes::TimeUnit::*;
use arrow::datatypes::{
    DataType::{self, *},
    Fields, Int32Type,
};

make_encoder!(EncodeBuilder {
    (Null, NullBuilder),
    (Bool, BooleanBuilder),
    // Number variants
    (I8, Int8Builder),
    (I16, Int16Builder),
    (I32, Int32Builder),
    (I64, Int64Builder),
    (UI8, UInt8Builder),
    (UI16, UInt16Builder),
    (UI32, UInt32Builder),
    (UI64, UInt64Builder),
    (F16, Float16Builder),
    (F32, Float32Builder),
    (F64, Float64Builder),

    // String variants
    (Str, StringBuilder),
    (LargeStr, LargeStringBuilder),
    (StrView, StringViewBuilder),

    // Timestamp variants
    (TsSecond, TimestampSecondBuilder),
    (TsMs, TimestampMillisecondBuilder),
    (TsMicro, TimestampMicrosecondBuilder),
    (TsNano, TimestampNanosecondBuilder),

    // Date variants
    (Date32, Date32Builder),
    (Date64, Date64Builder),

    // Time variants
    (Time32Second, Time32SecondBuilder),
    (Time32Ms, Time32MillisecondBuilder),
    (Time64Micro, Time64MicrosecondBuilder),
    (Time64Nano, Time64NanosecondBuilder),

    // Duration variants
    (DurationSecond, DurationSecondBuilder),
    (DurationMs, DurationMillisecondBuilder),
    (DurationMicro, DurationMicrosecondBuilder),
    (DurationNano, DurationNanosecondBuilder),

    // Interval variants
    (IntervalYearMonth, IntervalYearMonthBuilder),
    (IntervalDayTime, IntervalDayTimeBuilder),
    (IntervalMonthDayNano, IntervalMonthDayNanoBuilder),

    // Binary variants
    (Binary, BinaryBuilder),
    (FixedSizeBinary, FixedSizeBinaryBuilder),
    (LargeBinary, LargeBinaryBuilder),
    (BinaryView, BinaryViewBuilder),

    // // Decimal variants
    (Decimal32, Decimal32Builder),
    // (Decimal64, Decimal64Array),
    (Decimal128, Decimal128Builder),
    (Decimal256, Decimal256Builder),

    // List variants
    (List, ListBuilderWrapper),
    (FixedSizeList, FixedSizeListBuilderWrapper),
    (LargeList, LargeListBuilderWrapper),

    // Complex variants
    (Struct, StructBuilderWrapper),
    (Dictionary, DictionaryBuilderWrapper),
    (Map, MapBuilderWrapper),
});

pub struct ListBuilderWrapper {
    builder: ListBuilder<Box<EncodeBuilder>>,
    field: Arc<Field>,
}

impl ListBuilderWrapper {
    pub fn new(builder: ListBuilder<Box<EncodeBuilder>>, field: Arc<Field>) -> Self {
        Self { builder, field }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false);
    }

    pub fn values(&mut self) -> &mut Box<EncodeBuilder> {
        self.builder.values()
    }

    pub fn append(&mut self, is_valid: bool) -> Result<(), ArrowError> {
        self.builder.append(is_valid);
        Ok(())
    }
}

impl ArrayBuilder for ListBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }

    fn len(&self) -> usize {
        self.builder.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let list_array = self.builder.finish();
        let offsets = list_array.offsets().clone();
        let values = list_array.values().clone();
        let nulls = list_array.nulls().cloned();
        Arc::new(ListArray::new(self.field.clone(), offsets, values, nulls))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let list_array = self.builder.finish_cloned();
        let offsets = list_array.offsets().clone();
        let values = list_array.values().clone();
        let nulls = list_array.nulls().cloned();
        Arc::new(ListArray::new(self.field.clone(), offsets, values, nulls))
    }
}

pub struct FixedSizeListBuilderWrapper {
    builder: FixedSizeListBuilder<Box<EncodeBuilder>>,
    field: Arc<Field>,
    size: i32,
}

impl FixedSizeListBuilderWrapper {
    pub fn new(
        builder: FixedSizeListBuilder<Box<EncodeBuilder>>,
        field: Arc<Field>,
        size: i32,
    ) -> Self {
        Self {
            builder,
            field,
            size,
        }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false);
    }

    pub fn values(&mut self) -> &mut Box<EncodeBuilder> {
        self.builder.values()
    }

    pub fn append(&mut self, is_valid: bool) -> Result<(), ArrowError> {
        self.builder.append(is_valid);
        Ok(())
    }
}

impl ArrayBuilder for FixedSizeListBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }

    fn len(&self) -> usize {
        self.builder.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let array = self.builder.finish();
        let values = array.values().clone();
        let nulls = array.nulls().cloned();
        let size = self.size;
        Arc::new(FixedSizeListArray::new(
            self.field.clone(),
            size,
            values,
            nulls,
        ))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let array = self.builder.finish_cloned();
        let values = array.values().clone();
        let nulls = array.nulls().cloned();
        let size = self.size;
        Arc::new(FixedSizeListArray::new(
            self.field.clone(),
            size,
            values,
            nulls,
        ))
    }
}

// Wrapper for StructBuilder to work with EncodeBuilder field builders
pub struct StructBuilderWrapper {
    // builder: StructBuilder,
    fields: Fields,
    builders: Vec<Box<EncodeBuilder>>,
    null_buffer_builder: NullBufferBuilder,
}

impl ArrayBuilder for StructBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self as &mut dyn Any
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn len(&self) -> usize {
        self.null_buffer_builder.len()
    }

    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.finnish())
    }

    fn finish_cloned(&self) -> ArrayRef {
        Arc::new(self.finish_cloned())
    }
}

impl StructBuilderWrapper {
    pub fn new(fields: Fields, builders: Vec<Box<EncodeBuilder>>) -> Self {
        if fields.len() != builders.len() {
            panic!("Number of fields is not equal to the number of field_builders.");
        }
        Self {
            fields,
            builders,
            null_buffer_builder: NullBufferBuilder::new(0),
        }
    }

    pub fn append(&mut self, is_valid: bool) {
        self.null_buffer_builder.append(is_valid);
    }

    pub fn append_null(&mut self) {
        self.append(false);
    }

    pub fn finnish(&mut self) -> StructArray {
        if self.fields.is_empty() {
            return StructArray::new_empty_fields(self.len(), self.null_buffer_builder.finish());
        }

        let arrays = self.builders.iter_mut().map(|f| f.finish()).collect();
        let nulls = self.null_buffer_builder.finish();
        StructArray::new(self.fields.clone(), arrays, nulls)
    }

    pub fn finish_cloned(&self) -> StructArray {
        if self.fields.is_empty() {
            return StructArray::new_empty_fields(
                self.len(),
                self.null_buffer_builder.finish_cloned(),
            );
        }

        let arrays = self.builders.iter().map(|f| f.finish_cloned()).collect();
        let nulls = self.null_buffer_builder.finish_cloned();
        StructArray::new(self.fields.clone(), arrays, nulls)
    }
}

// Wrapper for LargeListBuilder
pub struct LargeListBuilderWrapper {
    builder: LargeListBuilder<Box<EncodeBuilder>>,
    field: Arc<Field>,
}

impl LargeListBuilderWrapper {
    pub fn new(builder: LargeListBuilder<Box<EncodeBuilder>>, field: Arc<Field>) -> Self {
        Self { builder, field }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false);
    }

    pub fn values(&mut self) -> &mut Box<EncodeBuilder> {
        self.builder.values()
    }

    pub fn append(&mut self, is_valid: bool) -> Result<(), ArrowError> {
        self.builder.append(is_valid);
        Ok(())
    }
}

impl ArrayBuilder for LargeListBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }
    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }
    fn len(&self) -> usize {
        self.builder.len()
    }
    fn finish(&mut self) -> ArrayRef {
        let list = self.builder.finish();
        let offsets = list.offsets().clone();
        let values = list.values().clone();
        let nulls = list.nulls().cloned();
        Arc::new(LargeListArray::new(
            self.field.clone(),
            offsets,
            values,
            nulls,
        ))
    }
    fn finish_cloned(&self) -> ArrayRef {
        let list = self.builder.finish_cloned();
        let offsets = list.offsets().clone();
        let values = list.values().clone();
        let nulls = list.nulls().cloned();
        Arc::new(LargeListArray::new(
            self.field.clone(),
            offsets,
            values,
            nulls,
        ))
    }
}

// Wrapper for Dictionary builders
pub struct DictionaryBuilderWrapper {
    builder: StringDictionaryBuilder<Int32Type>,
}

impl DictionaryBuilderWrapper {
    pub fn new(builder: StringDictionaryBuilder<Int32Type>) -> Self {
        Self { builder }
    }

    pub fn append_null(&mut self) {
        self.builder.append_null();
    }

    pub fn append_value(&mut self, value: &str) -> Result<(), ArrowError> {
        self.builder.append_value(value);
        Ok(())
    }
}

impl ArrayBuilder for DictionaryBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }
    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }
    fn len(&self) -> usize {
        self.builder.len()
    }
    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.builder.finish())
    }
    fn finish_cloned(&self) -> ArrayRef {
        Arc::new(self.builder.finish_cloned())
    }
}

// Wrapper for MapBuilder
pub struct MapBuilderWrapper {
    builder: MapBuilder<StringBuilder, Box<EncodeBuilder>>,
    field: Arc<Field>,
}

impl MapBuilderWrapper {
    pub fn new(builder: MapBuilder<StringBuilder, Box<EncodeBuilder>>, field: Arc<Field>) -> Self {
        Self { builder, field }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false).unwrap();
    }

    pub fn keys(&mut self) -> &mut StringBuilder {
        self.builder.keys()
    }

    pub fn values(&mut self) -> &mut Box<EncodeBuilder> {
        self.builder.values()
    }

    pub fn append(&mut self, is_valid: bool) -> Result<(), ArrowError> {
        self.builder.append(is_valid)
    }
}

impl ArrayBuilder for MapBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }
    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }
    fn len(&self) -> usize {
        self.builder.len()
    }
    fn finish(&mut self) -> ArrayRef {
        let map = self.builder.finish();
        let offsets = map.offsets().clone();
        let entries = map.entries();
        let nulls = map.nulls().cloned();
        let sorted = match self.field.data_type() {
            DataType::Map(_, s) => *s,
            _ => false,
        };
        let rebuilt_entries: StructArray = match self.field.data_type() {
            DataType::Map(entry_field, _) => match entry_field.data_type() {
                DataType::Struct(struct_fields) => {
                    let key = entries.column(0).clone();
                    let value = entries.column(1).clone();
                    StructArray::new(
                        struct_fields.clone(),
                        vec![key, value],
                        entries.nulls().cloned(),
                    )
                }
                _ => entries.clone(),
            },
            _ => entries.clone(),
        };
        Arc::new(MapArray::new(
            self.field.clone(),
            offsets,
            rebuilt_entries,
            nulls,
            sorted,
        ))
    }
    fn finish_cloned(&self) -> ArrayRef {
        let map = self.builder.finish_cloned();
        let offsets = map.offsets().clone();
        let entries = map.entries();
        let nulls = map.nulls().cloned();
        let sorted = match self.field.data_type() {
            DataType::Map(_, s) => *s,
            _ => false,
        };
        let rebuilt_entries: StructArray = match self.field.data_type() {
            DataType::Map(entry_field, _) => match entry_field.data_type() {
                DataType::Struct(struct_fields) => {
                    let key = entries.column(0).clone();
                    let value = entries.column(1).clone();
                    StructArray::new(
                        struct_fields.clone(),
                        vec![key, value],
                        entries.nulls().cloned(),
                    )
                }
                _ => entries.clone(),
            },
            _ => entries.clone(),
        };
        Arc::new(MapArray::new(
            self.field.clone(),
            offsets,
            rebuilt_entries,
            nulls,
            sorted,
        ))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EncodeBuilderError {
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(DataType),
    #[error("Arrow error: {0}")]
    ArrowError(#[from] ArrowError),
}

impl EncodeBuilder {
    pub fn try_new(data_type: &DataType) -> Result<Self, EncodeBuilderError> {
        match data_type {
            Null => Ok(EncodeBuilder::Null(NullBuilder::new())),
            Boolean => Ok(EncodeBuilder::Bool(BooleanBuilder::new())),
            Int8 => Ok(EncodeBuilder::I8(Int8Builder::new())),
            Int16 => Ok(EncodeBuilder::I16(Int16Builder::new())),
            Int32 => Ok(EncodeBuilder::I32(Int32Builder::new())),
            Int64 => Ok(EncodeBuilder::I64(Int64Builder::new())),
            UInt8 => Ok(EncodeBuilder::UI8(UInt8Builder::new())),
            UInt16 => Ok(EncodeBuilder::UI16(UInt16Builder::new())),
            UInt32 => Ok(EncodeBuilder::UI32(UInt32Builder::new())),
            UInt64 => Ok(EncodeBuilder::UI64(UInt64Builder::new())),
            Float16 => Ok(EncodeBuilder::F16(Float16Builder::new())),
            Float32 => Ok(EncodeBuilder::F32(Float32Builder::new())),
            Float64 => Ok(EncodeBuilder::F64(Float64Builder::new())),
            // Timestamp variants
            Timestamp(Second, _tz) => Ok(EncodeBuilder::TsSecond(TimestampSecondBuilder::new())),
            Timestamp(Millisecond, _tz) => {
                Ok(EncodeBuilder::TsMs(TimestampMillisecondBuilder::new()))
            }
            Timestamp(Microsecond, _tz) => {
                Ok(EncodeBuilder::TsMicro(TimestampMicrosecondBuilder::new()))
            }
            Timestamp(Nanosecond, _tz) => {
                Ok(EncodeBuilder::TsNano(TimestampNanosecondBuilder::new()))
            }

            // Date variants
            Date32 => Ok(EncodeBuilder::Date32(Date32Builder::new())),
            Date64 => Ok(EncodeBuilder::Date64(Date64Builder::new())),

            // Time variants
            Time32(Second) => Ok(EncodeBuilder::Time32Second(Time32SecondBuilder::new())),
            Time32(Millisecond) => Ok(EncodeBuilder::Time32Ms(Time32MillisecondBuilder::new())),
            Time64(Microsecond) => Ok(EncodeBuilder::Time64Micro(Time64MicrosecondBuilder::new())),
            Time64(Nanosecond) => Ok(EncodeBuilder::Time64Nano(Time64NanosecondBuilder::new())),

            // Duration variants
            Duration(Second) => Ok(EncodeBuilder::DurationSecond(DurationSecondBuilder::new())),
            Duration(Millisecond) => {
                Ok(EncodeBuilder::DurationMs(DurationMillisecondBuilder::new()))
            }
            Duration(Microsecond) => Ok(EncodeBuilder::DurationMicro(
                DurationMicrosecondBuilder::new(),
            )),
            Duration(Nanosecond) => {
                Ok(EncodeBuilder::DurationNano(DurationNanosecondBuilder::new()))
            }

            // Interval variants
            Interval(YearMonth) => Ok(EncodeBuilder::IntervalYearMonth(
                IntervalYearMonthBuilder::new(),
            )),
            Interval(DayTime) => Ok(EncodeBuilder::IntervalDayTime(IntervalDayTimeBuilder::new())),
            Interval(MonthDayNano) => Ok(EncodeBuilder::IntervalMonthDayNano(
                IntervalMonthDayNanoBuilder::new(),
            )),

            // Binary variants
            Binary => Ok(EncodeBuilder::Binary(BinaryBuilder::new())),
            FixedSizeBinary(size) => Ok(EncodeBuilder::FixedSizeBinary(
                FixedSizeBinaryBuilder::new(*size),
            )),
            LargeBinary => Ok(EncodeBuilder::LargeBinary(LargeBinaryBuilder::new())),
            BinaryView => Ok(EncodeBuilder::BinaryView(BinaryViewBuilder::new())),

            // String variants
            Utf8 => Ok(EncodeBuilder::Str(StringBuilder::new())),
            LargeUtf8 => Ok(EncodeBuilder::LargeStr(LargeStringBuilder::new())),
            Utf8View => Ok(EncodeBuilder::StrView(StringViewBuilder::new())),

            // List variants
            List(field) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder = ListBuilder::new(inner_builder);
                Ok(EncodeBuilder::List(ListBuilderWrapper::new(
                    list_builder,
                    field.clone(),
                )))
            }

            // Decimal variants - use Decimal128Builder for smaller decimals
            Decimal32(precision, scale) => Ok(EncodeBuilder::Decimal32(
                Decimal32Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),
            Decimal128(precision, scale) => Ok(EncodeBuilder::Decimal128(
                Decimal128Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),
            Decimal256(precision, scale) => Ok(EncodeBuilder::Decimal256(
                Decimal256Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),

            // Additional List variants
            FixedSizeList(field, size) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder = FixedSizeListBuilder::new(inner_builder, *size);
                Ok(EncodeBuilder::FixedSizeList(
                    FixedSizeListBuilderWrapper::new(list_builder, field.clone(), *size),
                ))
            }
            LargeList(field) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder = LargeListBuilder::new(inner_builder);
                Ok(EncodeBuilder::LargeList(LargeListBuilderWrapper::new(
                    list_builder,
                    field.clone(),
                )))
            }

            // Struct variant
            Struct(fields) => {
                let field_builders: Vec<Box<EncodeBuilder>> = fields
                    .iter()
                    .map(|f| EncodeBuilder::try_new(f.data_type()).map(Box::new))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(EncodeBuilder::Struct(StructBuilderWrapper::new(
                    fields.clone(),
                    field_builders,
                )))
            }

            // Dictionary variant
            Dictionary(_key_type, value_type) => match value_type.as_ref() {
                Utf8 => {
                    let dict_builder = StringDictionaryBuilder::<Int32Type>::new();
                    Ok(EncodeBuilder::Dictionary(DictionaryBuilderWrapper::new(
                        dict_builder,
                    )))
                }
                _ => Err(EncodeBuilderError::UnsupportedDataType(data_type.clone())),
            },

            // Map variant
            Map(field, _sorted) => {
                if let Struct(map_fields) = field.data_type() {
                    if map_fields.len() == 2 {
                        let key_field = &map_fields[0];
                        let value_field = &map_fields[1];

                        let field_names = MapFieldNames {
                            entry: field.name().to_string(),
                            key: key_field.name().to_string(),
                            value: value_field.name().to_string(),
                        };

                        let key_builder = StringBuilder::new();
                        let value_builder =
                            Box::new(EncodeBuilder::try_new(value_field.data_type())?);
                        let map_builder =
                            MapBuilder::new(Some(field_names), key_builder, value_builder);

                        Ok(EncodeBuilder::Map(MapBuilderWrapper::new(
                            map_builder,
                            field.clone(),
                        )))
                    } else {
                        Err(EncodeBuilderError::UnsupportedDataType(data_type.clone()))
                    }
                } else {
                    Err(EncodeBuilderError::UnsupportedDataType(data_type.clone()))
                }
            }

            dt => Err(EncodeBuilderError::UnsupportedDataType(dt.clone())),
        }
    }

    pub fn try_with_capacity(
        data_type: &DataType,
        capacity: usize,
    ) -> Result<Self, EncodeBuilderError> {
        match data_type {
            Null => Ok(EncodeBuilder::Null(NullBuilder::new())),
            Boolean => Ok(EncodeBuilder::Bool(BooleanBuilder::new())),
            Int8 => Ok(EncodeBuilder::I8(Int8Builder::new())),
            Int16 => Ok(EncodeBuilder::I16(Int16Builder::new())),
            Int32 => Ok(EncodeBuilder::I32(Int32Builder::new())),
            Int64 => Ok(EncodeBuilder::I64(Int64Builder::new())),
            UInt8 => Ok(EncodeBuilder::UI8(UInt8Builder::new())),
            UInt16 => Ok(EncodeBuilder::UI16(UInt16Builder::new())),
            UInt32 => Ok(EncodeBuilder::UI32(UInt32Builder::new())),
            UInt64 => Ok(EncodeBuilder::UI64(UInt64Builder::new())),
            Float16 => Ok(EncodeBuilder::F16(Float16Builder::new())),
            Float32 => Ok(EncodeBuilder::F32(Float32Builder::new())),
            Float64 => Ok(EncodeBuilder::F64(Float64Builder::new())),
            // Timestamp variants
            Timestamp(Second, _tz) => Ok(EncodeBuilder::TsSecond(TimestampSecondBuilder::new())),
            Timestamp(Millisecond, _tz) => {
                Ok(EncodeBuilder::TsMs(TimestampMillisecondBuilder::new()))
            }
            Timestamp(Microsecond, _tz) => {
                Ok(EncodeBuilder::TsMicro(TimestampMicrosecondBuilder::new()))
            }
            Timestamp(Nanosecond, _tz) => {
                Ok(EncodeBuilder::TsNano(TimestampNanosecondBuilder::new()))
            }

            // Date variants
            Date32 => Ok(EncodeBuilder::Date32(Date32Builder::new())),
            Date64 => Ok(EncodeBuilder::Date64(Date64Builder::with_capacity(
                capacity,
            ))),

            // Time variants
            Time32(Second) => Ok(EncodeBuilder::Time32Second(Time32SecondBuilder::new())),
            Time32(Millisecond) => Ok(EncodeBuilder::Time32Ms(Time32MillisecondBuilder::new())),
            Time64(Microsecond) => Ok(EncodeBuilder::Time64Micro(Time64MicrosecondBuilder::new())),
            Time64(Nanosecond) => Ok(EncodeBuilder::Time64Nano(Time64NanosecondBuilder::new())),

            // Duration variants
            Duration(Second) => Ok(EncodeBuilder::DurationSecond(DurationSecondBuilder::new())),
            Duration(Millisecond) => {
                Ok(EncodeBuilder::DurationMs(DurationMillisecondBuilder::new()))
            }
            Duration(Microsecond) => Ok(EncodeBuilder::DurationMicro(
                DurationMicrosecondBuilder::new(),
            )),
            Duration(Nanosecond) => {
                Ok(EncodeBuilder::DurationNano(DurationNanosecondBuilder::new()))
            }

            // Interval variants
            Interval(YearMonth) => Ok(EncodeBuilder::IntervalYearMonth(
                IntervalYearMonthBuilder::new(),
            )),
            Interval(DayTime) => Ok(EncodeBuilder::IntervalDayTime(IntervalDayTimeBuilder::new())),
            Interval(MonthDayNano) => Ok(EncodeBuilder::IntervalMonthDayNano(
                IntervalMonthDayNanoBuilder::new(),
            )),

            // Binary variants
            Binary => Ok(EncodeBuilder::Binary(BinaryBuilder::with_capacity(
                capacity, 1024,
            ))),
            FixedSizeBinary(size) => Ok(EncodeBuilder::FixedSizeBinary(
                FixedSizeBinaryBuilder::with_capacity(capacity, *size),
            )),
            LargeBinary => Ok(EncodeBuilder::LargeBinary(
                LargeBinaryBuilder::with_capacity(capacity, 1024),
            )),
            BinaryView => Ok(EncodeBuilder::BinaryView(BinaryViewBuilder::with_capacity(
                capacity,
            ))),

            // String variants
            Utf8 => Ok(EncodeBuilder::Str(StringBuilder::with_capacity(
                capacity, 1024,
            ))),
            LargeUtf8 => Ok(EncodeBuilder::LargeStr(LargeStringBuilder::with_capacity(
                capacity, 1024,
            ))),
            Utf8View => Ok(EncodeBuilder::StrView(StringViewBuilder::with_capacity(
                capacity,
            ))),

            // List variants
            List(field) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder = ListBuilder::with_capacity(inner_builder, capacity);
                Ok(EncodeBuilder::List(ListBuilderWrapper::new(
                    list_builder,
                    field.clone(),
                )))
            }

            // Additional List variants
            FixedSizeList(field, size) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder =
                    FixedSizeListBuilder::with_capacity(inner_builder, *size, capacity);
                Ok(EncodeBuilder::FixedSizeList(
                    FixedSizeListBuilderWrapper::new(list_builder, field.clone(), *size),
                ))
            }
            LargeList(field) => {
                let inner_builder = Box::new(EncodeBuilder::try_with_capacity(
                    field.data_type(),
                    capacity,
                )?);
                let list_builder = LargeListBuilder::with_capacity(inner_builder, capacity);
                Ok(EncodeBuilder::LargeList(LargeListBuilderWrapper::new(
                    list_builder,
                    field.clone(),
                )))
            }

            // Decimal variants - same as try_new since they don't use capacity
            Decimal32(precision, scale) => Ok(EncodeBuilder::Decimal32(
                Decimal32Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),
            Decimal128(precision, scale) => Ok(EncodeBuilder::Decimal128(
                Decimal128Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),
            Decimal256(precision, scale) => Ok(EncodeBuilder::Decimal256(
                Decimal256Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),

            // Struct variant
            Struct(fields) => {
                let field_builders: Vec<Box<EncodeBuilder>> = fields
                    .iter()
                    .map(|f| {
                        EncodeBuilder::try_with_capacity(f.data_type(), capacity).map(Box::new)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(EncodeBuilder::Struct(StructBuilderWrapper::new(
                    fields.clone(),
                    field_builders,
                )))
            }

            // Dictionary variant - use capacity
            Dictionary(_key_type, value_type) => match value_type.as_ref() {
                Utf8 => {
                    let dict_builder = StringDictionaryBuilder::<Int32Type>::with_capacity(
                        capacity, capacity, capacity,
                    );
                    Ok(EncodeBuilder::Dictionary(DictionaryBuilderWrapper::new(
                        dict_builder,
                    )))
                }
                _ => Err(EncodeBuilderError::UnsupportedDataType(data_type.clone())),
            },

            // Map variant - use capacity
            Map(field, _sorted) => {
                if let Struct(map_fields) = field.data_type() {
                    if map_fields.len() == 2 {
                        let key_field = &map_fields[0];
                        let value_field = &map_fields[1];

                        // Create field names that match the schema
                        let field_names = MapFieldNames {
                            entry: field.name().to_string(),
                            key: key_field.name().to_string(),
                            value: value_field.name().to_string(),
                        };

                        let key_builder = StringBuilder::with_capacity(capacity, capacity * 10);
                        let value_builder = Box::new(EncodeBuilder::try_with_capacity(
                            value_field.data_type(),
                            capacity,
                        )?);
                        let map_builder = MapBuilder::with_capacity(
                            Some(field_names),
                            key_builder,
                            value_builder,
                            capacity,
                        );

                        Ok(EncodeBuilder::Map(MapBuilderWrapper::new(
                            map_builder,
                            field.clone(),
                        )))
                    } else {
                        Err(EncodeBuilderError::UnsupportedDataType(data_type.clone()))
                    }
                } else {
                    Err(EncodeBuilderError::UnsupportedDataType(data_type.clone()))
                }
            }

            dt => Err(EncodeBuilderError::UnsupportedDataType(dt.clone())),
        }
    }
}
