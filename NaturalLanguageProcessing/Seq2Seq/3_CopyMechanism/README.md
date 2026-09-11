# Pointer-Generator Copy Mechanism

← [Seq2Seq](../README.md)

`PointerGenerator` combines a vocabulary distribution with an attention-derived
copy distribution over source positions. This supports target tokens that are
rare or absent from a fixed vocabulary. The custom layer and model builder are
implemented in TensorFlow; dataset wiring is currently a scaffold.
