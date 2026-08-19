"""COMPINT: the offline benchmark arm.

Spec 20.1: may import `shared`. May NOT import `scguard`. A CI check enforces this, because
research code drifting into production only behavior would silently invalidate reproduction.
"""
