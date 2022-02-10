# linear_equation_solver
Objective:
Project 2
The aim of this assignment is to compare and analyze the behavior of numerical
methods studied in class: Gaussian-elimination, LU decomposition, Gaussian-Jordan
and Gauss-Seidel.
Description:
You are required to implement a program forsolving systems of linear equations, which
takes as an input the equations, the technique to use and its required parameters.
Specification:
- The program must contain the following features:
1- An interactive GUI that enables the user to enter a set of linear equations.
2- Reading from files must be available as well (all the inputs are available in the same
file).
3- A way to choose a method to solve the given equation (Preferably a drop down list
or buttons), also a way to choose to use all the methods and provide text boxes to
enter the parameters for each method.
4- A way to enter the precision and the max number of iterations otherwise default
values are used, Default Max Iterations = 50, Default Epsilon = 0.00001;
5- The answer for the chosen method indicating the number of iterations (if exists),
execution time, all iterations’ approximate root and precision.
6- You need to output the above results in a file preferably in a tabular format.
7- In the case of using an iterative method, you need to plot the following curve for
every variable separately: -
i. Curve between the number of iterations and the obtained root value at this
iteration for all the methods in the same graph
