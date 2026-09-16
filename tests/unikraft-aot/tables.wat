(module
  (type $value_type (func (result i32)))
  (table 4 8 funcref)
  (func $value (type $value_type) i32.const 42)
  (elem (i32.const 0) funcref (ref.func $value) (ref.null func))
  (elem $passive funcref (ref.null func) (ref.func $value))
  (func (export "is_null") (param i32) (result i32)
    local.get 0
    table.get 0
    ref.is_null)
  (func (export "indirect") (param i32) (result i32)
    local.get 0
    call_indirect (type $value_type))
  (func (export "init") (param i32 i32 i32)
    local.get 0
    local.get 1
    local.get 2
    table.init $passive)
  (func (export "drop") elem.drop $passive))
