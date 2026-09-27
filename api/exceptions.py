class AnalyticsError(Exception):
    """Base exception for analytics-related errors."""


class AnalyticsDataNotFoundError(AnalyticsError):
    """Raised when required analytics data is missing."""


class AnalyticsQueryError(AnalyticsError):
    """Raised when an analytics query fails."""