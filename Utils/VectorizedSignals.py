import numpy as np

class VectorizedSignals:
    """Helper class for safe vectorized signal operations"""

    @staticmethod
    def safe_signals(long_condition, short_condition):
        """
        Safely generate signals from long and short conditions.
        Handles NaN values automatically.

        Args:
            long_condition: Boolean series for long entries
            short_condition: Boolean series for short entries

        Returns:
            np.array: Signal array with 1 (long), -1 (short), 0 (no signal)
        """
        long_safe = (
            long_condition.fillna(False)
            if hasattr(long_condition, "fillna")
            else long_condition
        )
        short_safe = (
            short_condition.fillna(False)
            if hasattr(short_condition, "fillna")
            else short_condition
        )
        return np.where(long_safe, 1, np.where(short_safe, -1, 0))

    @staticmethod
    def safe_condition(condition):
        """Make any condition safe by filling NaN with False"""
        return condition.fillna(False) if hasattr(condition, "fillna") else condition
