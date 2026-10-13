from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import exception_handler


class DealerDeskAPIException(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "VALIDATION_ERROR"
    default_message = "Validation error"

    def __init__(self, message=None, code=None, details=None, status_code=None):
        if status_code is not None:
            self.status_code = status_code
        self.code = code or self.default_code
        self.message = message or self.default_message
        self.details = details if details is not None else []
        super().__init__(detail=self.message, code=self.code)


class ValidationError(DealerDeskAPIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "VALIDATION_ERROR"


class NotFoundError(DealerDeskAPIException):
    status_code = status.HTTP_404_NOT_FOUND
    default_code = "NOT_FOUND"
    default_message = "Resource not found"


class DuplicateError(DealerDeskAPIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "DUPLICATE"
    default_message = "Resource already exists"


class InsufficientStockError(DealerDeskAPIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "INSUFFICIENT_STOCK"
    default_message = "Insufficient stock"


class InvalidTransitionError(DealerDeskAPIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "INVALID_TRANSITION"
    default_message = "Invalid status transition"


class OrderLockedError(DealerDeskAPIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "ORDER_LOCKED"
    default_message = "Order is locked and cannot be modified"


class IdempotencyConflictError(DealerDeskAPIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "IDEMPOTENCY_CONFLICT"
    default_message = "Idempotency key reused with different request parameters"


class SupplierUnavailableError(DealerDeskAPIException):
    status_code = status.HTTP_502_BAD_GATEWAY
    default_code = "SUPPLIER_UNAVAILABLE"
    default_message = "Supplier feed unavailable"


def custom_exception_handler(exc, context):
    """
    Standard DRF exception handler wrapped to ensure exact contract error envelope:
    {"error": {"code": "...", "message": "...", "details": [...]}}
    """
    if isinstance(exc, DealerDeskAPIException):
        return Response(
            {
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
            status=exc.status_code,
        )

    # Let standard DRF exception handler handle other exceptions (e.g. built-in 404, 400, etc.)
    response = exception_handler(exc, context)

    if response is not None:
        status_code = response.status_code
        code = "VALIDATION_ERROR"
        if status_code == 404:
            code = "NOT_FOUND"
        elif status_code == 409:
            code = "DUPLICATE"
        elif status_code == 502:
            code = "SUPPLIER_UNAVAILABLE"

        # Build message and details
        message = ""
        details = []

        if isinstance(response.data, dict):
            if "detail" in response.data:
                message = str(response.data["detail"])
            else:
                field_errors = []
                for field, errs in response.data.items():
                    err_msg = errs if isinstance(errs, str) else "; ".join(str(e) for e in errs)
                    field_errors.append(f"{field}: {err_msg}")
                    details.append({"field": field, "error": err_msg})
                message = "; ".join(field_errors) if field_errors else "Validation error"
        elif isinstance(response.data, list):
            message = "; ".join(str(e) for e in response.data)
        else:
            message = str(response.data)

        if not message:
            message = "An error occurred"

        return Response(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "details": details,
                }
            },
            status=status_code,
        )

    return None
