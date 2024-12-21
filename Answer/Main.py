from PageRank import *
from WeightedPageRank import *


######################################### Test the functions for PageRank file :
def test_PageRank():
    print(construct_link_matrix(), "\n")
    eigenvalues, eigenvectors, _, _ = solve_eigen_problem(construct_link_matrix())
    print("eigenvalues : \n" , eigenvalues)
    print("eigenvectors : \n" , eigenvectors)
    display_search_results()
    display_search_results_with_plot()


######################################### Test the functions for WeightedPageRank file :
def test_Weighted_PageRank():
    print(construct_weight_matrix(), "\n")
    eigenvalues, eigenvectors, _, _ = solve_eigen_problem_for_weight_matrix(construct_weight_matrix())
    print("eigenvalues : \n" , eigenvalues)
    print("eigenvectors : \n" , eigenvectors)
    display_search_results_for_weight_matrix()
    display_search_results_with_plot_for_weight_matrix()


##### Run the tests
test_PageRank()
# test_Weighted_PageRank()