import matplotlib
matplotlib.use('Agg')  # Use Agg backend
from home.models import Contact
from django.http import HttpResponseBadRequest
from django.shortcuts import render, HttpResponse,redirect, HttpResponseRedirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.contrib import messages
from django.urls import reverse
import csv
from io import StringIO
import io
import base64
import os
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import zipfile
from matplotlib.lines import Line2D

@login_required(login_url='login')
def home(request):
    return render(request,'home.html')
    # return HttpResponse("This is homepage")

def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        # Check if any required field is empty
        if not uname or not email or not pass1 or not pass2:
            return render(request, 'signup.html', {'error': "Please fill in all fields"})

        if pass1 != pass2:
            return render(request, 'signup.html', {'error': "Password and confirm password do not match"})
        else:
            try:
                existing_user = User.objects.get(username=uname)
                return render(request, 'signup.html', {'error': "Username already exists"})
            except User.DoesNotExist:
                pass
            
            try:
                existing_email = User.objects.get(email=email)
                return render(request, 'signup.html', {'error': "Email already exists"})
            except User.DoesNotExist:
                pass

            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()
            return redirect('login')

    return render(request, 'signup.html')


def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('pass')

        # Check for empty username or password
        if not username or not password:
            messages.error(request, "Please enter both username and password.")
            return render(request, 'login.html')

        try:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                # Username or password is incorrect
                messages.error(request, "Invalid username or password.")
        except (ValueError, ValidationError) as e:  # Catch potential validation errors
            messages.error(request, "There was an error during login. Please try again.")
        except Exception as e:  # Catch unexpected errors (log for debugging)
            print(f"Unexpected error during login: {e}")
            messages.error(request, "An unexpected error occurred. Please try again later.")

    return render(request, 'login.html')



def LogoutPage(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def about(request):
    return render(request,'about.html')

    # return HttpResponse("This is aboutpage")

@login_required(login_url='login')
def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        reason = request.POST.get('reason')

        if not (name and email and phone and reason):
            messages.error(request, "Please fill in all fields.")
        else:
            contact = Contact.objects.create(name=name, email=email, phone=phone, reason=reason)
            messages.success(request, "Your message has been submitted successfully!")

    return render(request, 'contact.html')
    # return HttpResponse("This is contactpage")
# Define your clean_data function here

def clean_data(df):
    # Perform data cleaning operations here (e.g., removing null values, formatting columns)
    cleaned_df = df.dropna()  # Example: Removing rows with missing values
    return cleaned_df

@login_required(login_url='login')
def upload(request):
    file_name = None
    cleaned_data = None
    cleaned_data_full = None  # Variable to store the full cleaned dataset

    if request.method == 'POST':
        if 'dataFile' in request.FILES:
            file = request.FILES['dataFile']
            file_name = file.name
            if file_name.endswith('.csv') or file_name.endswith('.xlsx') or file_name.endswith('.txt'):
                df = pd.read_csv(file) if file_name.endswith('.csv') else pd.read_excel(file) if file_name.endswith('.xlsx') else pd.read_csv(file, delimiter='\t')
                messages.success(request, 'File uploaded successfully.')
                cleaned_df = clean_data(df)
                cleaned_data_full = cleaned_df.to_html()  # Store entire cleaned DataFrame

                cleaned_data = cleaned_df.head(10).to_html()
                request.session['cleaned_data'] = cleaned_data
                request.session['cleaned_data_full'] = cleaned_data_full             
            else:
                messages.error(request, 'Invalid file format. Please upload a .csv, .xlsx, or .txt file.')
                return redirect('upload')
        else:
            messages.error(request, 'Please upload a file before running analysis.')
            return redirect('upload')

        if 'analysis_type' in request.POST:
            analysis_types = request.POST.getlist('analysis_type')
            request.session['selected_analysis_types'] = analysis_types

            for analysis_type in analysis_types:
                if analysis_type == 'descriptive':
                    if 'cleaned_df' in locals():
                        descriptive_stats = cleaned_df.describe().to_html()
                        request.session['descriptive_stats'] = descriptive_stats
                    else:
                        messages.error(request, 'Please upload a file before running analysis.')
                        return redirect('upload')

                elif analysis_type == 'regression':
                    if 'cleaned_df' in locals():
                        # Drop non-numeric columns
                        numeric_df = cleaned_df.select_dtypes(include=[np.number])
                        regression_result = perform_regression_analysis(numeric_df)
                        if regression_result:
                            request.session['regression_result'] = regression_result
                        else:
                            messages.error(request, 'Insufficient data for regression analysis. Please ensure the dataset has at least two numeric columns.')
                            return redirect('upload')
                    else:
                        messages.error(request, 'Please upload a file before running analysis.')
                        return redirect('upload')
                    
                elif analysis_type == 'correlation':
                    if 'cleaned_df' in locals():
                        correlation_result = perform_correlation_analysis(cleaned_df)
                        if correlation_result is not None:
                            request.session['correlation_result'] = correlation_result
                        else:
                            messages.error(request, 'Insufficient data for correlation analysis.')
                            return redirect('upload')
                    else:
                        messages.error(request, 'Please upload a file before running analysis.')
                        return redirect('upload')

                elif analysis_type == 'outlier':
                    if 'cleaned_df' in locals():
                        outlier_result = perform_outlier_detection(cleaned_df)
                        request.session['outlier_result'] = outlier_result
                    else:
                        messages.error(request, 'Please upload a file before running analysis.')
                        return redirect('upload')
                    
                elif analysis_type == 'histogram':
                    if 'cleaned_df' in locals():
                        histogram_plots = generate_histogram_plots(cleaned_df)
                        request.session['histogram_plots'] = histogram_plots
                    else:
                        messages.error(request, 'Please upload a file before running analysis.')
                        return redirect('upload')
                    
                elif analysis_type == 'scatter':
                    if 'cleaned_df' in locals():
                        scatter_plots = generate_scatter_plots(cleaned_df)
                        request.session['scatter_plots'] = scatter_plots
                    else:
                        messages.error(request, 'Please upload a file before running analysis.')
                        return redirect('upload')
                    
                elif analysis_type == 'boxplot':
                    if 'cleaned_df' in locals():
                        boxplot_plots = generate_boxplot_plots(cleaned_df)
                        request.session['boxplot_plots'] = boxplot_plots
                    else:
                        messages.error(request, 'Please upload a file before running analysis.')
                        return redirect('upload')
                
                elif analysis_type == 'pairplot':
                    if 'cleaned_df' in locals():
                        pair_plots = generate_pair_plots(cleaned_df)
                        # Modify the keys to replace underscores with ' vs '
                        pair_plots = {pair.replace('_', ' vs '): plot for pair, plot in pair_plots.items()}
                        request.session['pair_plots'] = pair_plots
                    else:
                        messages.error(request, 'Please upload a file before running analysis.')
                        return redirect('upload')


                # elif analysis_type == 'clustering':
                #     if 'cleaned_df' in locals():
                #         clustering_result = perform_clustering_analysis(cleaned_df)
                #         if clustering_result is not None:
                #             request.session['clustering_result'] = clustering_result
                #         else:
                #             messages.error(request, 'Insufficient data for clustering analysis.')
                #             return redirect('upload')
                #     else:
                #         messages.error(request, 'Please upload a file before running analysis.')
                #         return redirect('upload')

            return redirect('analysis_result')

    else:  # Assuming initial GET request
        cleaned_data = None  # Initialize cleaned_data to None

    return render(request, 'upload.html', {
        'file_name': file_name,  # Set file_name if available
        'cleaned_data': cleaned_data,
    })

def perform_regression_analysis(cleaned_df):
    regression_results = {}
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype in [np.int64, np.float64]:  # Check if column contains numerical data
            X = cleaned_df.drop(columns=[column])  # Independent variables
            y = cleaned_df[column]  # Dependent variable

            model = LinearRegression()
            model.fit(X, y)

            regression_results[column] = {
                'Coefficients': model.coef_.tolist(),  # Convert NumPy array to Python list
                'Intercept': model.intercept_,
                # Add more regression metrics if needed
            }

    return regression_results

def perform_correlation_analysis(cleaned_df):
    correlation_results = {}

    # Numerical correlation analysis
    numerical_df = cleaned_df.select_dtypes(include=[np.number])
    if not numerical_df.empty and len(numerical_df.columns) > 1:
        numerical_correlation_matrix = numerical_df.corr()
        correlation_results['numerical_correlation_matrix'] = numerical_correlation_matrix.to_html()

    # Categorical correlation analysis
    categorical_df = cleaned_df.select_dtypes(exclude=[np.number])
    if not categorical_df.empty:
        label_encoder = LabelEncoder()
        categorical_encoded = categorical_df.apply(label_encoder.fit_transform)
        categorical_correlation_matrix = categorical_encoded.corr(method='pearson')
        correlation_results['categorical_correlation_matrix'] = categorical_correlation_matrix.to_html()

    return correlation_results

# def perform_clustering_analysis(cleaned_df):
#     # Perform clustering analysis and return the result
#     # For example, let's perform KMeans clustering
#     kmeans = KMeans(n_clusters=3)
#     cluster_labels = kmeans.fit_predict(cleaned_df)
#     cleaned_df['Cluster'] = cluster_labels
    
#     # Convert the result to a dictionary
#     clustering_result = cleaned_df.to_dict(orient='list')
    
#     return clustering_result

def perform_outlier_detection(cleaned_df):
    # Implement outlier detection algorithm (e.g., Z-score, IQR)
    # Here's a simple example using Z-score
    outlier_result = {}
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype in [np.int64, np.float64]:
            z_scores = (cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std()
            outliers = cleaned_df[abs(z_scores) > 3]  # Threshold for outliers (e.g., Z-score > 3)
            outlier_result[column] = outliers.to_html()
    return outlier_result


def generate_histogram_plots(cleaned_df):
    # Define custom colors for histograms
    custom_colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'lightcoral', 'cyan', 'gold', 'pink', 'lightblue']
    
    # Generate histogram plots
    histogram_plots = {}
    for i, column in enumerate(cleaned_df.columns):
        if cleaned_df[column].dtype in [np.int64, np.float64]:
            plt.figure(figsize=(8, 6))
            sns.histplot(cleaned_df[column], kde=True, color=custom_colors[i % len(custom_colors)])  # Use custom color
            plt.title(f'Histogram of {column}', fontsize=16)
            plt.xlabel(column, fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'histogram_{column}.png')
            plt.close()
            with open(f'histogram_{column}.png', 'rb') as f:
                plot_bytes = base64.b64encode(f.read()).decode('utf-8')
            histogram_plots[column] = plot_bytes
    return histogram_plots

def generate_scatter_plots(cleaned_df):
    # Define custom colors for scatter plots
    custom_colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'lime']
    
    # Generate scatter plots
    scatter_plots = {}
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for i, (column_x, column_y) in enumerate(itertools.combinations(numeric_columns, 2)):
        plt.figure(figsize=(8, 6))
        plt.scatter(cleaned_df[column_x], cleaned_df[column_y], color=custom_colors[i % len(custom_colors)], alpha=0.6)  # Use custom color
        plt.title(f'Scatter plot of {column_x} vs {column_y}', fontsize=16)
        plt.xlabel(column_x, fontsize=14)
        plt.ylabel(column_y, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'scatter_{column_x}_{column_y}.png')
        plt.close()
        with open(f'scatter_{column_x}_{column_y}.png', 'rb') as f:
            plot_bytes = base64.b64encode(f.read()).decode('utf-8')
        scatter_plots[f'{column_x}_{column_y}'] = plot_bytes
    return scatter_plots

def generate_boxplot_plots(cleaned_df):
    # Define custom colors for boxplots
    custom_colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'lightcoral', 'cyan', 'gold', 'pink', 'lightblue']
    
    # Generate boxplot plots with Seaborn
    boxplot_plots = {}
    for i, column in enumerate(cleaned_df.columns):
        if cleaned_df[column].dtype in [np.int64, np.float64]:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=cleaned_df[column], color=custom_colors[i % len(custom_colors)], linewidth=2.5)
            plt.title(f'Boxplot of {column}', fontsize=16)
            plt.xlabel(column, fontsize=14)
            plt.ylabel('Value', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'boxplot_{column}.png')
            plt.close()
            with open(f'boxplot_{column}.png', 'rb') as f:
                plot_bytes = base64.b64encode(f.read()).decode('utf-8')
            boxplot_plots[column] = plot_bytes
    return boxplot_plots


def generate_pair_plots(cleaned_df):
    # Generate pair plots
    pair_plots = {}
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    pair_combinations = list(itertools.combinations(numeric_columns, 2))  # Generate combinations of numeric columns

    for pair in pair_combinations:
        plt.figure(figsize=(8, 6))
        sns.set(style="whitegrid")  # Set seaborn style to 'whitegrid'
        pair_plot = sns.pairplot(cleaned_df, vars=pair, kind='scatter', diag_kind='kde', plot_kws={'alpha': 0.6})
        pair_plot.fig.suptitle(f'Pair Plot of {pair[0]} vs {pair[1]}', fontsize=16, ha='center')  # Centered title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to leave space for title
        
        # Customize colors and markers
        color_palette = sns.color_palette("hsv", len(pair_combinations))  # Use a different color palette, e.g., 'hsv'
        for i, ax in enumerate(pair_plot.axes.flat):
            if i % (len(pair_combinations) + 1) == 0:  # Diagonal plots
                ax.set_facecolor('lightgray')
            else:
                ax.set_facecolor(color_palette[i % len(pair_combinations)])
                if isinstance(ax.collections, list):
                    for collection in ax.collections:
                        collection.set_color(color_palette[i % len(pair_combinations)])
                        collection.set_alpha(0.6)  # Adjust transparency

        # Adjust linewidth of KDE plots
        for diag_ax in np.diag(pair_plot.axes):
            for line in diag_ax.get_lines():
                line.set_linewidth(2)

        pair_plot.savefig(f'pairplot_{pair[0]}_{pair[1]}.png')
        plt.close()

        with open(f'pairplot_{pair[0]}_{pair[1]}.png', 'rb') as f:
            plot_bytes = base64.b64encode(f.read()).decode('utf-8')
        pair_plots[f'{pair[0]}_{pair[1]}'] = plot_bytes

    return pair_plots

def analysis_result(request):
    cleaned_data = request.session.get('cleaned_data', None)
    selected_analysis_types = request.session.get('selected_analysis_types', [])

    if not cleaned_data:
        messages.error(request, 'No analysis result found. Please upload a file and select analysis types first.')
        return redirect('upload')

    descriptive_stats = None
    regression_output = None
    correlation_output = None
    histogram_plots = None  # Initialize histogram_plots
    scatter_plots = None  # Initialize histogram_plots
    boxplot_plots = None  # Initialize histogram_plots
    pair_plots_formatted = None
    outlier_output = None


    # clustering_output = None

    if 'descriptive' in selected_analysis_types:
        descriptive_stats = request.session.get('descriptive_stats', None)
        if not descriptive_stats:
            descriptive_stats = generate_descriptive_stats(cleaned_data)
            request.session['descriptive_stats'] = descriptive_stats

    if 'regression' in selected_analysis_types:
        regression_result = request.session.get('regression_result', None)
        if not regression_result:
            regression_result = perform_regression_analysis(cleaned_data)
            request.session['regression_result'] = regression_result
        regression_output = generate_regression_output(regression_result)

    if 'correlation' in selected_analysis_types:
        correlation_result = request.session.get('correlation_result', None)
        if not correlation_result:
            correlation_result = perform_correlation_analysis(cleaned_data)
            request.session['correlation_result'] = correlation_result
        correlation_output = generate_correlation_output(correlation_result) if correlation_result else None
    else:
        correlation_output = None

    if 'outlier' in selected_analysis_types:  # Check for outlier detection analysis
        outlier_result = request.session.get('outlier_result', None)
        if outlier_result:
            outlier_output = generate_outlier_output(outlier_result)
        else:
            messages.error(request, 'Outlier detection analysis result not found.')
            return redirect('upload')

    if 'histogram' in selected_analysis_types:  # Check for histogram analysis
        histogram_plots = request.session.get('histogram_plots', None)
        if not histogram_plots:
            histogram_plots = generate_histogram_plots(cleaned_data)
            request.session['histogram_plots'] = histogram_plots

    if 'scatter' in selected_analysis_types:
        scatter_plots = request.session.get('scatter_plots', None)
        if not scatter_plots:
            scatter_plots = generate_scatter_plots(cleaned_data)
            request.session['scatter_plots'] = scatter_plots

    if 'boxplot' in selected_analysis_types:
        boxplot_plots = request.session.get('boxplot_plots', None)
        if not boxplot_plots:
            boxplot_plots = generate_boxplot_plots(cleaned_data)  # Implement this function
            request.session['boxplot_plots'] = boxplot_plots

    if 'pairplot' in selected_analysis_types:
        pair_plots = request.session.get('pair_plots', None)
        if not pair_plots:
            pair_plots = generate_pair_plots(cleaned_data)
            pair_plots_formatted = {pair.replace('_', ' vs '): plot for pair, plot in pair_plots.items()}
            request.session['pair_plots'] = pair_plots
        else:
            pair_plots_formatted = {pair.replace('_', ' vs '): plot for pair, plot in pair_plots.items()}
    else:
        pair_plots_formatted = None  # Reset pair_plots_formatted if 'pairplot' analysis type is not selected


    # if 'clustering' in selected_analysis_types:
    #     clustering_result = request.session.get('clustering_result', None)
    #     if not clustering_result:
    #         num_clusters = 3  # You can change this value as needed
    #         clustering_result = perform_clustering_analysis(cleaned_data, num_clusters)
    #     clustering_output = generate_clustering_output(clustering_result)

    return render(request, 'analysis_result.html', {
        'cleaned_data': cleaned_data,
        'descriptive_stats': descriptive_stats,
        'regression_output': regression_output,
        'correlation_output': correlation_output,
        'outlier_output': outlier_output,
        'histogram_plots': histogram_plots,  # Pass histogram_plots to the template
        'scatter_plots': scatter_plots,  # Pass histogram_plots to the template
        'boxplot_plots': boxplot_plots,  # Pass histogram_plots to the template
        'pair_plots': pair_plots_formatted,

        # 'clustering_output': clustering_output,
    })

def generate_descriptive_stats(cleaned_data):
    # Convert the HTML table of cleaned data back to a DataFrame
    cleaned_df = pd.read_html(cleaned_data)[0]

    # Generate descriptive statistics for numeric columns
    numeric_stats = cleaned_df.describe().to_html()

    # Generate descriptive statistics for categorical columns
    categorical_stats = cleaned_df.describe(include='object').to_html()

    # Combine the statistics for numeric and categorical columns
    descriptive_stats = f"<h3>Numeric Variables:</h3>{numeric_stats}<br><h3>Categorical Variables:</h3>{categorical_stats}"

    return descriptive_stats

def generate_regression_output(regression_result):
    output = ""
    output += "<table border='1'>"
    output += "<tr><th>Variable</th><th>Coefficients</th><th>Intercept</th></tr>"
    for column, result in regression_result.items():
        output += f"<tr><td>{column}</td><td>{', '.join(map(str, result['Coefficients']))}</td><td>{result['Intercept']}</td></tr>"
    output += "</table>"
    return output

def generate_correlation_output(correlation_result):
    output = ""
    for matrix_name, matrix_data in correlation_result.items():
        output += f"<h4>{matrix_name}</h4>"
        output += matrix_data
    return output

def generate_outlier_output(outlier_result):
    output = ""
    for column, outliers in outlier_result.items():
        if isinstance(outliers, pd.DataFrame) and not outliers.empty:
            output += f"<h4>{column}</h4>"
            output += outliers.to_html()
        elif isinstance(outliers, str) and outliers.strip():  # Check if string is not empty
            output += f"<h4>{column}</h4>"
            output += f"<p>{outliers}</p>"
    return output

# def generate_clustering_output(clustering_result):
#     output = "<h3>Clustering Analysis Result:</h3>"
#     output += "<table border='1'>"
#     output += "<tr><th>Index</th>"
#     for key in clustering_result.keys():
#         output += f"<th>{key}</th>"
#     output += "</tr>"
    
#     for i in range(len(clustering_result['Cluster'])):
#         output += f"<tr><td>{i}</td>"
#         for key in clustering_result.keys():
#             output += f"<td>{clustering_result[key][i]}</td>"
#         output += "</tr>"
    
#     output += "</table>"
    
#     return output

def download_results(request):
    cleaned_data_full = request.session.get('cleaned_data_full')  # Fetch full cleaned dataset

    if cleaned_data_full:
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="cleaned_data.csv"'
        df = pd.read_html(cleaned_data_full)[0]  # Convert HTML table back to DataFrame
        df.to_csv(path_or_buf=response, index=False)
        return response
    else:
        return HttpResponse("No data available for download.")

def download_descriptive_stats(request):
    descriptive_stats = request.session.get('descriptive_stats')  # Fetch descriptive statistics

    if descriptive_stats:
        # Parse HTML table and extract data
        table_data = pd.read_html(descriptive_stats)[0]
        
        # Create a StringIO buffer to write CSV content
        csv_buffer = StringIO()
        
        # Write the data to the StringIO buffer in CSV format
        table_data.to_csv(csv_buffer, index=False)
        
        # Create the HTTP response with CSV content
        response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="descriptive_stats.csv"'
        
        return response
    else:
        return HttpResponse("No descriptive statistics available for download.")

def download_regression_results(request):
    regression_results = request.session.get('regression_result')  # Fetch regression results from session

    if regression_results:
        # Create a CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="regression_results.csv"'

        # Write regression results to CSV
        writer = csv.writer(response)
        writer.writerow(['Variable', 'Coefficients', 'Intercept'])  # Header row

        for variable, result in regression_results.items():
            coefficients = ', '.join(map(str, result['Coefficients']))
            writer.writerow([variable, coefficients, result['Intercept']])

        return response
    else:
        return HttpResponse("No regression results available for download.")

def download_correlation_results(request):
    correlation_result = request.session.get('correlation_result')  # Fetch correlation analysis result from session

    if correlation_result:
        # Create a CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="correlation_results.csv"'

        # Write correlation analysis result to CSV
        for matrix_name, matrix_data in correlation_result.items():
            # Parse HTML table and extract numerical values
            df = pd.read_html(matrix_data, header=0, index_col=0)[0]
            # Write DataFrame to CSV
            df.to_csv(response, header=True)

            # Add a separator between matrices
            response.write('\n\n')

        return response
    else:
        return HttpResponse("No correlation analysis results available for download.")

def download_outlier_results(request):
    outlier_result = request.session.get('outlier_result')  # Fetch outlier analysis result from session

    if outlier_result:
        # Create a CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="outlier_results.csv"'

        # Write outlier analysis result to CSV
        for column, outliers in outlier_result.items():
            if isinstance(outliers, pd.DataFrame) and not outliers.empty:
                # Write column name as header
                response.write(f"{column}\n")
                # Write DataFrame to CSV
                outliers.to_csv(response, index=False)
                response.write('\n')
            elif isinstance(outliers, str) and outliers.strip():  # Check if string is not empty
                # Write column name as header
                response.write(f"{column}\n")
                # Write outlier string to CSV
                response.write(outliers)
                response.write('\n\n')

        return response
    else:
        return HttpResponse("No outlier results available for download.")

def download_histogram_plots(request):
    histogram_plots = request.session.get('histogram_plots')  # Fetch histogram plots from session

    if histogram_plots:
        # Create a ZIP file containing all histogram plots
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for column, plot_bytes in histogram_plots.items():
                # Decode base64 encoded image bytes
                image_bytes = base64.b64decode(plot_bytes)
                # Add image file to ZIP archive
                zip_file.writestr(f'{column}_histogram.png', image_bytes)

        # Prepare response
        response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="histogram_plots.zip"'
        
        return response
    else:
        return HttpResponse("No histogram plots available for download.")

def download_scatter_plots(request):
    scatter_plots = request.session.get('scatter_plots')  # Fetch scatter plots from session

    if scatter_plots:
        # Create a ZIP file containing all scatter plots
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for plot_name, plot_bytes in scatter_plots.items():
                # Decode base64 encoded image bytes
                image_bytes = base64.b64decode(plot_bytes)
                # Add image file to ZIP archive
                zip_file.writestr(f'{plot_name}.png', image_bytes)

        # Prepare response
        response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="scatter_plots.zip"'
        
        return response
    else:
        return HttpResponse("No scatter plots available for download.")

def download_boxplot_plots(request):
    boxplot_plots = request.session.get('boxplot_plots')  # Fetch box plot plots from session

    if boxplot_plots:
        # Create a ZIP file containing all box plot plots
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for column, plot_bytes in boxplot_plots.items():
                # Decode base64 encoded image bytes
                image_bytes = base64.b64decode(plot_bytes)
                # Add image file to ZIP archive
                zip_file.writestr(f'{column}_boxplot.png', image_bytes)

        # Prepare response
        response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="boxplot_plots.zip"'
        
        return response
    else:
        return HttpResponse("No box plot plots available for download.")

def download_pair_plots(request):
    pair_plots = request.session.get('pair_plots')  # Fetch pair plots from session

    if pair_plots:
        # Create a ZIP file containing all pair plots
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for plot_name, plot_bytes in pair_plots.items():
                # Decode base64 encoded image bytes
                image_bytes = base64.b64decode(plot_bytes)
                # Add image file to ZIP archive
                zip_file.writestr(f'{plot_name}.png', image_bytes)

        # Prepare response
        response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="pair_plots.zip"'
        
        return response
    else:
        return HttpResponse("No pair plots available for download.")

