import os
import re
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from PageRank import *


################################ Section 1 ################################
def parse_page_links_with_weights():
    output = dict()
    pages_dir = "WeightedPages"

    if not os.path.exists(pages_dir):
        raise FileNotFoundError(f"Directory '{pages_dir}' does not exist!")

    page_pattern = re.compile(r'(Page\d+).html')
    link_pattern = re.compile(r'link:to:(Page\d+):(\d\.\d+)')

    for file_name in os.listdir(pages_dir):
        file_path = os.path.join(pages_dir, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                page_name = page_pattern.findall(file_name)[0]
                links = link_pattern.findall(content)
                output[page_name] = [(link, float(weight)) for link, weight in links]
    
    return output


############################## Section 2 ###################################
def construct_weight_matrix():
    extracted_links = parse_page_links_with_weights()
    num_pages = len(extracted_links)
    link_matrix = np.zeros((num_pages, num_pages))
    
    for page_name, links_of_page in extracted_links.items():
        page_index = int(page_name.split('Page')[1]) - 1
        total_weight_of_page = sum(weight for _, weight in links_of_page)
        if links_of_page and total_weight_of_page > 0:
            for linked_page, weight in links_of_page:
                linked_page_index = int(linked_page.split('Page')[1]) - 1
                link_matrix[linked_page_index, page_index] = weight * (1 / total_weight_of_page)        #### Normalize weights
        else:
            link_matrix[:, page_index] = (1 / num_pages)
    
    return link_matrix


############################## Section 3 ###################################
def solve_eigen_problem_for_weight_matrix(matrix):
    #### Find the characteristic polynomial:
    n = matrix.shape[0]
    identity_matrix = np.eye(n)
    characteristic_polynomial = np.poly(matrix)
    eigenvalues = [np.abs(ev) for ev in np.roots(characteristic_polynomial)]

    #### Find the eigenvectors:
    eigenvectors = []
    eigenvalue_of_steady_state = None
    eigenvector_of_steady_state = None
    for eigenvalue in eigenvalues:
        characteristic_matrix = matrix - eigenvalue * identity_matrix
        eigenvector = null_space(characteristic_matrix)

        if eigenvector.shape[1] != 0:                   #### Check if the eigenvector is not null
            if np.isclose(eigenvalue, 1):               #### Check for steady state eigenvalue
                eigenvalue_of_steady_state = eigenvalue
                eigenvector_of_steady_state = eigenvector[:, 0]
            eigenvectors.append(eigenvector[:, 0])

    eigenvectors = np.array(eigenvectors).T
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] /= np.sum(eigenvectors[:, i])

    if eigenvector_of_steady_state is not None:
        eigenvector_of_steady_state /= np.sum(eigenvector_of_steady_state)

    return eigenvalues, eigenvectors, eigenvalue_of_steady_state, eigenvector_of_steady_state


################################# Section 4 ######################################
def power_method_for_weight_matrix(matrix):
    result = []
    num_simulations = 1000
    epsilon = 10**(-6)
    """ Power Iteration to find the dominant eigenvalue and eigenvector. 
    Parameters: 
        matrix (numpy.array): The input matrix 
        num_simulations (int): Number of iterations for convergence 
    Returns: 
        tuple: Dominant eigenvalue and corresponding eigenvector """ 
    n = matrix.shape[0] 
    b_k = np.random.rand(n) 
    b_k = b_k / np.linalg.norm(b_k, 1)

    for _ in range(num_simulations): 
        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(matrix, b_k)
        # Re normalize the vector 
        b_k1_norm = np.linalg.norm(b_k1, 1) 
        b_k1 /= b_k1_norm
        # convergence and stopping criteria
        if np.linalg.norm(b_k1 - b_k, 1) < epsilon:
            break
        b_k = b_k1
    # Rayleigh quotient to approximate the dominant eigenvalue 
    eigenvalue = np.dot(b_k.T, np.dot(matrix, b_k)) / np.dot(b_k.T, b_k) 
    result = (eigenvalue, b_k)
    
    return result 


########################################## Section 5 #####################################
def display_search_results_for_weight_matrix(search_request = None):
    matrix = construct_weight_matrix()
    eigenvalue1, eigenvector1 = power_method_for_weight_matrix(matrix)
    _, _, eigenvalue2, eigenvector2 = solve_eigen_problem_for_weight_matrix(matrix)

    if search_request:
        search_results_power_method = [
            (i, eigenvector1[i]) for i in np.argsort(eigenvector1)[::-1] if f"Page {i+1}".lower() in search_request.lower()
            ] 
        search_results_solve_eigen_problem = [
            (i, eigenvector2[i]) for i in np.argsort(eigenvector2)[::-1] if f"Page {i+1}".lower() in search_request.lower()
            ] 
    else: 
        search_results_power_method = [(i, eigenvector1[i]) for i in np.argsort(eigenvector1)[::-1]] 
        search_results_solve_eigen_problem = [(i, eigenvector2[i]) for i in np.argsort(eigenvector2)[::-1]]

    print("\nPower Method Results for eigenvalue: ", eigenvalue1)
    print("\n| Grade | Page ID | PageRank |")
    print("\n|-------|---------|----------|")
    for (page, rank) in search_results_power_method:
        print(f"\n|   {list(np.argsort(eigenvector1)[::-1]).index(page) + 1}   | Page {page + 1}  |  {rank:.4f}  |")
        print("\n|-------|---------|----------|")
    print("\n")

    print("\nSearch Results for eigenvalue: ", eigenvalue2)
    print("\n| Grade | Page ID | PageRank |")
    print("\n|-------|---------|----------|")
    for (page, rank) in search_results_solve_eigen_problem:
        print(f"\n|   {list(np.argsort(eigenvector2)[::-1]).index(page) + 1}   | Page {page + 1}  |  {rank:.4f}  |")
        print("\n|-------|---------|----------|")


###################################### Section 6 ###################################
def display_search_results_with_plot_for_weight_matrix():
    #### Create the matrix :
    matrix = construct_weight_matrix()
    eigenvalue1, eigenvector1 = power_method_for_weight_matrix(matrix)
    _, _, eigenvalue2, eigenvector2 = solve_eigen_problem_for_weight_matrix(matrix)

    matrix = construct_link_matrix()
    _, eigenvector_without_weight1 =  power_method(matrix)
    _, _, _, eigenvector_without_weight2 = solve_eigen_problem(matrix)

    ranks = [i + 1 for i in range(eigenvector1.shape[0])]
    #### Sorted the value of that :
    sorted_eigenvector_power_method = [(i, eigenvector1[i]) for i in np.argsort(eigenvector1)[::-1]]
    pages_power_method = [f"Page {page + 1}" for (page, _) in sorted_eigenvector_power_method]
    grants_power_method = [grant for (_, grant) in sorted_eigenvector_power_method]

    sorted_power_method = [(i, eigenvector_without_weight1[i]) for i in np.argsort(eigenvector_without_weight1)[::-1]]
    pages_p = [f"Page {page + 1}" for (page, _) in sorted_power_method]
    sorted_grants_p = [grant for (_, grant) in sorted_power_method]
    ranks_p = [pages_p.index(page) + 1 for page in pages_power_method]
    grants_p = [sorted_grants_p[rank -1] for rank in ranks_p]

    sorted_eigenvector_solve_eigen_problem = [(i, eigenvector2[i]) for i in np.argsort(eigenvector2)[::-1]]
    pages_solve_eigen = [f"Page {page + 1}" for (page, _) in sorted_eigenvector_solve_eigen_problem]
    grants_solve_eigen = [grant for (_, grant) in sorted_eigenvector_solve_eigen_problem]

    sorted_solve_eigen_problem = [(i, eigenvector_without_weight2[i]) for i in np.argsort(eigenvector_without_weight2)[::-1]]
    pages_s = [f"Page {page + 1}" for (page, _) in sorted_solve_eigen_problem]
    sorted_grants_s = [grant for (_, grant) in sorted_solve_eigen_problem]
    ranks_s = [pages_s.index(page) + 1 for page in pages_solve_eigen]
    grants_s = [sorted_grants_s[rank -1] for rank in ranks_s]

    #### Display the plot :   
    plt.figure(figsize=(14, 7))
    bar_width = 0.2
    ##### Figure for Power Method :
    plt.subplot(1, 2, 1)
    index = np.arange(len(pages_power_method))
    bar1 = plt.bar(index, grants_power_method, bar_width, color='blue', label='grant of Pages')
    bar2 = plt.bar(index + bar_width, grants_p, bar_width, color='red', label='grant without Weights')
    bar3 = plt.bar(index + 2 * bar_width, ranks, bar_width, color='green', label='Rank of Pages')
    bar4 = plt.bar(index + 3 * bar_width, ranks_p, bar_width, color='yellow', label='Rank without Weights')
    plt.xlabel('Pages')
    plt.ylabel('Value of Pages')
    plt.title(f'Power Method Result figure for eigenvalue {eigenvalue1}')
    plt.xticks(index + bar_width / 2, pages_power_method, rotation= 45)
    plt.legend()
    #### Figure for Eigen Problem :
    plt.subplot(1, 2, 2)
    index = np.arange(len(pages_solve_eigen))
    bar1 = plt.bar(index, grants_solve_eigen, bar_width, color='blue', label='grant of Pages') 
    bar3 = plt.bar(index + bar_width, grants_s, bar_width, color='red', label='grant without Weights')
    bar3 = plt.bar(index + 2 * bar_width, ranks, bar_width, color='green', label='Rank of Pages')
    bar4 = plt.bar(index + 3 * bar_width, ranks_s, bar_width, color='yellow', label='Rank without Weights')
    plt.xlabel('Pages')
    plt.ylabel('Value of Pages')
    plt.title(f'Eigen Problem Result figure for eigenvalue {eigenvalue2}')
    plt.xticks(index + bar_width / 2, pages_solve_eigen, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()