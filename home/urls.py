from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [
    path("", views.SignupPage, name='signup'),
    path("login/", views.LoginPage, name='login'),
    path('logout/',views.LogoutPage,name='logout'),
    path("home/", views.home, name='home'),
    path("about/", views.about, name='about'),
    # path("services/", views.services, name='services'),
    path("contact/", views.contact, name='contact'),
    path("upload/", views.upload, name='upload'),
    path('analysis_result/', views.analysis_result, name='analysis_result'),
    path('download/', views.download_results, name='download_results'),
    path('download_descriptive_stats/', views.download_descriptive_stats, name='download_descriptive_stats'),
    path('download_regression_results/', views.download_regression_results, name='download_regression_results'),
    path('download_correlation_results/', views.download_correlation_results, name='download_correlation_results'),
    path('download_outlier_results/', views.download_outlier_results, name='download_outlier_results'),
    path('download_histogram_plots/', views.download_histogram_plots, name='download_histogram_plots'),
    path('download_scatter_plots/', views.download_scatter_plots, name='download_scatter_plots'),
    path('download_boxplot_plots/', views.download_boxplot_plots, name='download_boxplot_plots'),
    path('download_pair_plots/', views.download_pair_plots, name='download_pair_plots'),
]