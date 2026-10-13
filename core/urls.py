from django.urls import path

from core import views

urlpatterns = [
    path("health/", views.health, name="health"),
    # Products
    path("products/", views.product_list, name="product-list"),
    path("products/<str:sku>/", views.product_detail, name="product-detail"),
    # Inventory
    path("inventory/<str:sku>/", views.inventory_detail, name="inventory-detail"),
    # Dealers
    path("dealers/", views.dealer_list, name="dealer-list"),
    path("dealers/<int:pk>/", views.dealer_detail, name="dealer-detail"),
    # Orders
    path("orders/", views.order_list, name="order-list"),
    path("orders/<str:order_number>/", views.order_detail, name="order-detail"),
    path("orders/<str:order_number>/confirm/", views.order_confirm, name="order-confirm"),
    path("orders/<str:order_number>/dispatch/", views.order_dispatch, name="order-dispatch"),
    path("orders/<str:order_number>/deliver/", views.order_deliver, name="order-deliver"),
    path("orders/<str:order_number>/cancel/", views.order_cancel, name="order-cancel"),
    # Forecasting & Reorder
    path("forecast/<str:sku>/", views.forecast_detail, name="forecast-detail"),
    path("reorder-suggestions/", views.reorder_suggestions, name="reorder-suggestions"),
    # Assistant
    path("assistant/chat/", views.assistant_chat, name="assistant-chat"),
    # Supplier Sync
    path("sync/supplier/", views.sync_supplier, name="sync-supplier"),
]
