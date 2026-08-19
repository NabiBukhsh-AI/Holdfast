"""Code that MUST be identical between the research arm and the production arm.

Spec section 20.1: this is how INV-5 is guaranteed structurally rather than by convention.
`compint` and `scguard` may both import this package. Neither may import the other.
"""
