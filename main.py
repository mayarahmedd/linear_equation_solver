import re
import time
import tkinter
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfile

import matplotlib.pyplot
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.pyplot import subplots
from sympy import sympify, lambdify, poly, var


def disable(x):
    x["state"] = DISABLED


def enable(x):
    x["state"] = NORMAL


def resetCanvasAndToolbar():
    if 'canvas' in globals():
        canvas.get_tk_widget().place_forget()
        matplotlib.pyplot.cla()


def getVariables(n):
    variables = []
    for i in range(int(n)):
        variables = variables + re.findall("[a-z]", entries[i].get())
    variables = sorted(list(dict.fromkeys(variables)))
    return variables


def convertToList(n, variables):
    res = []
    for i in range(len(entries)):
        temp = poly(entries[i].get(), var(','.join(variables)))
        temp = temp.coeffs()
        for j in range(len(variables)):
            if entries[i].get().find(variables[j]) == -1:
                temp.insert(j, 0)
        if len(temp) == n:
            temp.insert(n + 1, 0)
        res = res + temp
    return res


def plot(prec):
    if not prec or len(
            entries) == 0 or methodnames.current() == 0 or methodnames.current() == 1 or methodnames.current() == 2:
        return
    for i in range(int(entry_eqNum.get())):
        if len(init_guesses[i].get()) == 0:
            root_label.config(text="You should enter guesses")
            root_label.grid(row=2, column=5)
            return None
    resetCanvasAndToolbar()
    eachIter.clear()
    n = int(entry_eqNum.get())
    variables = getVariables(n)
    tem = convertToList(n, variables)
    a = np.zeros((n, n + 1))

    # Making numpy array of n size and initializing
    # to zero for storing solution vector
    x = np.zeros(n)
    setMatrix(a, tem, n)
    old_x = [None] * n
    ea = [None] * n
    for i in range(n):
        eachIter.append([])
    x, ea, iternum = gaussSiedel(a, setInitials(n), n, float(prec), int(iters.get()), ea, old_x)
    t = len(eachIter[0])
    temp = []
    for i in range(t):
        temp.append(i + 1)

    for i in eachIter:
        plot1.plot(temp, i)
    plot1.legend(variables)
    plot1.set_ylabel("Roots")
    plot1.set_xlabel("Iterations")
    fig.tight_layout()

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().place(x=850, y=80)


def setInitials(n):
    x_temp = []
    if (len(lb) != 0):
        x_temp = []
        for i in range(n):
            if len(init_guesses[i].get()) == 0:
                root_label.config(text="You should enter guesses")
                root_label.grid(row=2, column=5)
                return None
            x_temp.append(float(init_guesses[i].get()))
    return x_temp


def submitOutput(scanner, prec, iter, result, variables):
    resetCanvasAndToolbar()
    disable(plot_button)
    if methodnames.current() == 3 or methodnames.current() == 4:
        enable(plot_button)
    iters.setvar(value=1)
    try:
        if len(prec) == 0:
            prec = 0.00001
        if len(iter) == 0:
            iter = 50
        j = 0
        n = int(entry_eqNum.get())
        a = np.zeros((n, n + 1))
        s = ""
        # Making numpy array of n size and initializing
        # to zero for storing solution vector
        x = np.zeros(n)
        setMatrix(a, result, n)
        start = time.perf_counter()
        ea = []
        if scanner == 0:
            x = gaussElimination(a, x, n)
        elif scanner == 1:
            x = LUdecomposition(a, x, n)
        elif scanner == 2:
            x = gaussJordan(a, x, n)
        elif scanner == 3:
            old_x = [None] * n
            ea = [None] * n
            eachIter.clear()
            eachEa.clear()
            j = 0
            for i in range(n):
                eachIter.append([])
                eachEa.append([])
                eachEa[j].append("_")
                j = j + 1
            for i in range(n):
                if len(init_guesses[i].get()) == 0:
                    root_label.config(text="You should enter guesses")
                    root_label.grid(row=2, column=5)
                    return None
            x, ea, iternum = gaussSiedel(a, setInitials(n), n, float(prec), int(iter), ea, old_x)
        elif scanner == 4:
            x = gaussElimination(a, x, n)
            s = "Gauss Elimination\n"
            for i in variables:
                s += i + "=" + "%f" % x[j] + "\n"
                j += 1

            s += "\n-----------------------------------------------------------------------\nLU Decomposition\n"
            j = 0
            x = np.zeros(n)
            setMatrix(a, result, n)
            x = LUdecomposition(a, x, n)
            for i in variables:
                s += i + "=" + "%f" % x[j] + "\n"
                j += 1

            s += "\n-----------------------------------------------------------------------\nGauss Jordan\n"
            j = 0
            x = np.zeros(n)
            setMatrix(a, result, n)
            x = gaussJordan(a, x, n)
            for i in variables:
                s += i + "=" + "%f" % x[j] + "\n"
                j += 1

            s += "\n-----------------------------------------------------------------------\nGauss Seidel\n"
            j = 0
            x = np.zeros(n)
            setMatrix(a, result, n)
            old_x = [None] * n
            ea = [None] * n
            eachIter.clear()
            eachEa.clear()

            for i in range(n):
                eachIter.append([])
                eachEa.append([])
                eachEa[i].append("_")

            for i in range(n):
                if len(init_guesses[i].get()) == 0:
                    root_label.config(text="You should enter guesses")
                    root_label.grid(row=2, column=5)
                    return None
            x, ea, iternum = gaussSiedel(a, setInitials(n), n, float(prec), int(iter), ea, old_x)
        if x is None:
            return root_str
        end = time.perf_counter()
        j = 0
        if scanner == 3 or scanner == 4:
            for i in variables:
                t = "\t{0:20}".format(str(x[j]))
                s += i + "=" + t + "\t, ea = " + str(ea[j]) + "\n"
                j += 1
        else:
            for i in variables:
                t = "\t{0:20}".format(str(x[j]))
                s += i + "=" + t + "\n"
                j += 1
        if scanner == 4 or scanner == 3:
            s += "Iterations = " + str(len(eachIter[0])-1) + ", "
        s += "Execution time = " + str(end - start) + "\n" + "Precision = " + prec
        iters.setvar(value=1)
        root_label.config(text=s)
        f = open("output.txt", "w")
        j = 0

        if scanner == 3 or scanner == 4:
            s += "\nGauss Seidel Tracing:"
            s += "\nIter\t"

            for i in variables:
                s += "{0:20}".format(i) + "\t"
            for i in variables:
                s += "ea(" + i + ")                  "
            s += "\n"
            if len(eachIter) != 0:
                for i in range(len(eachIter[0])):
                    s += str(i)
                    for j in range(len(eachIter)):
                        s += "\t{0:20}".format(str(eachIter[j][i]))
                    for j in range(len(eachEa)):
                        s += "\t{0:20}".format(str(eachEa[j][i]))
                    s += "\n"
        f.write(s)
        f.close()
    except Exception as e:
        root_label.config(text="Error occurred !")
        root_label.grid(row=2, column=5)
        print(e)
    root_label.place(x=450, y=100)
    return root_str


def validate_equation(inp):
    try:
        expr = sympify(inp)
        lambdify('x', expr)
    except:
        return False
    return True


def validate_float(inp):
    try:
        float(inp)
    except:
        return False
    return float(inp) >= 0


def validate_int(inp):
    try:
        int(inp)
    except:
        return False
    return int(inp) > 0


def get_combo_box(event):
    global row
    resetCanvasAndToolbar()
    scanner = event.widget.current()
    root_label.config(text="")
    if len(lb) != 0:
        for i in range(old_n):
            lb[i].destroy()
            init_guesses[i].destroy()
        lb.clear()
        init_guesses.clear()

    disable(plot_button)
    if (scanner == 3 or scanner == 4) and entry_eqNum.get() != "":
        enable(plot_button)
        row = getRow() + 1
        n = int(entry_eqNum.get())
        temp_eq = []
        for i in range(len(entries)):
            temp_eq.append(entries[i].get())
        initiateGUI(row)
        for i in range(len(temp_eq)):
            entries[i].insert(0, temp_eq[i])


def gaussElimination(a, x, n):
    # Forward Elimination
    for i in range(n):
        if a[i][i] == 0.0:
            root_label.config(text="Division by zero")
            root_label.grid(row=2, column=5)
            return None

        for j in range(i + 1, n):
            ratio = a[j][i] / a[i][i]

            for k in range(n + 1):  #to form the upper triangular matrix
                a[j][k] = a[j][k] - ratio * a[i][k]   #R2' = R2 - R1*ratio

    # Back Substitution
    x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = a[i][n]  #result (last column in augmented matrix) --> Matrix b

        for j in range(i + 1, n):
            x[i] = x[i] - a[i][j] * x[j]  #coeff*X2 = result[2] - X3*coeff

        x[i] = x[i] / a[i][i]  #divide by the coefficient
    return x


def gaussJordan(a, x, n):
    for i in range(n):
        if a[i][i] == 0.0:
            root_label.config(text="Division by zero")
            root_label.grid(row=2, column=5)
            return None

        temp = a[i][i]
        for norm in range(n + 1):  #Normalize the current row
            a[i][norm] = a[i][norm] / temp

        for j in range(n):  #to loop on all rows each time
            if i == j:  #to skip the current row
                continue
            else:
                ratio = a[j][i]  # a[i][i] = 1 (so, no need to divide)

                for k in range(n + 1):  #eliminate
                    a[j][k] = a[j][k] - ratio * a[i][k]

    for i in range(n):  #answer is the last column
        x[i] = a[i][n]

    return x


def LUdecomposition(a, x, n):
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            U[i][j] = a[i][j]  #U is the same as a but without answers (a --> augmented matrix,  U --> not augmented)
            if i == j:
                L[i][j] = 1  #diagonals = 1
            else:
                L[i][j] = 0

    # Forward Elimination
    for i in range(n):
        if U[i][i] == 0.0:
            root_label.config(text="Division by zero")
            root_label.grid(row=2, column=5)
            return None

        for j in range(i + 1, n):
            ratio = U[j][i] / U[i][i]
            L[j][i] = ratio  #lower triangular matrix containing ratios

            for k in range(n):  #eliminate (to construct an upper triangular matrix)
                U[j][k] = U[j][k] - ratio * U[i][k]

    # Forward Substitution  (L * y = b)
    y = np.zeros(n)
    y[0] = a[0][n]  #answer in the augmented matrix

    for i in range(1, n):
        y[i] = a[i][n]  #results

        for j in range(i):
            y[i] = y[i] - L[i][j] * y[j]  #There is no coeff. division as diagonals = 1 in L

    # Back Substitution (U * x = y)
    x[n - 1] = y[n - 1] / U[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = y[i]  #results

        for j in range(i + 1, n):
            x[i] = x[i] - U[i][j] * x[j]

        x[i] = x[i] / U[i][i]
    return x


def getRow():
    global row
    row = row + 1
    return row


def scanner(n):
    result = []
    variables = getVariables(n)
    if len(variables) != n:
        root_label.config(text="Number of variables are not equal to number of equations")
        root_label.grid(row=2, column=5)
        return

    result = convertToList(n, variables)
    submitOutput(methodnames.current(), prec.get(), iters.get(), result, variables)


def gaussSiedel(a, x, n, prec, iter, ea, old_x, iter_num=1):
    global iterVar
    for i in range(n):
        old_x[i] = x[i] # old_x is used to calculate ea
        eachIter[i].append(x[i]) #eachIter is used for plotting and writing in file
        iterVar = iterVar + 1

    for j in range(n):
        # temp variable b to store b[j]
        b = a[j][n] #result

        for i in range(n):
            if (j != i):
                b -= a[j][i] * x[i]  #subtract all row elements in the row from b (except the diagonal element)

        x[j] = b / a[j][j]  #divide by diagonal element
    for i in range(n):
        ea[i] = abs((x[i] - old_x[i]) / x[i])
        eachEa[i].append(ea[i])  #to be used in file

    if max(ea) <= prec or iter == 1:
        for i in range(n):
            eachIter[i].append(x[i])
            iterVar = iterVar + 1
        return x, ea, iter_num
    else:
        return gaussSiedel(a, x, n, prec, iter - 1, ea, old_x, iter_num + 1)


def setMatrix(a, res, n):
    m = 0
    for i in range(n):
        for j in range(n + 1):
            a[i][j] = int(res[m])
            m += 1
            if j == n:
                a[i][j] = -1 * a[i][j]


old_n = 0


def initiateGUI(row):
    if len(entry_eqNum.get()) != 0:
        root_label.config(text="")
        root_label.grid(row=2, column=5)
        n = int(entry_eqNum.get())
        global old_n
        global submit
        if len(labels) != 0:
            submit.destroy()
            for i in range(old_n):
                labels[i].destroy()
                entries[i].destroy()
                if len(lb) != 0:
                    lb[i].destroy()
                    init_guesses[i].destroy()
            labels.clear()
            entries.clear()
            if len(lb) != 0:
                lb.clear()
                init_guesses.clear()

        row = row + 1

        if methodnames.current() == 3 or methodnames.current() == 4:
            for i in range(n):
                label = tkinter.Label(window, text="Guess" + str(i + 1), bg=color, foreground=fg_color)
                label.grid(row=row, column=0)
                lb.append(label)
                entry = tkinter.Entry(window, validate='key')
                entry.grid(row=row, column=1)
                init_guesses.append(entry)
                row = row + 1

        for i in range(n):
            label = tkinter.Label(window, text="Equation" + str(i + 1), bg=color, foreground=fg_color)
            label.grid(row=row, column=0)
            labels.append(label)
            entry = tkinter.Entry(window, validate='key')
            entry.grid(row=row, column=1)
            entries.append(entry)
            row = row + 1

        old_n = n
        submit = Button(master=window,
                        command=lambda: scanner(n),
                        height=2,
                        width=10,
                        text="Submit",
                        bg="yellow")
        submit.grid(row=row, column=0, pady=10)
    else:
        root_label.config(text="Please enter number of equations")
        root_label.grid(row=2, column=5)
        return


def readFile():
    try:
        file = askopenfile(mode='r', filetypes=[('Text Files', '*.txt')])
        if file is not None:
            root_label.config(text="")
            fileContent = [line.rstrip() for line in file]
            n = fileContent[0]
            method = fileContent[1]

            entry_eqNum.delete(0, len(entry_eqNum.get()))
            entry_eqNum.insert(0, n)

            length = len(fileContent)
            if method == "Gaussian-elimination":
                methodnames.current(0)
            elif method == "LU-decomposition":
                methodnames.current(1)
            elif method == "Gaussian-jordan":
                methodnames.current(2)
            elif method == "Gaussian-seidel" or method == "All Methods":
                if method == "Gaussian-seidel":
                    methodnames.current(3)
                else:
                    methodnames.current(4)
                length = length - 1
                x_initials = []
                x_initials = fileContent[length].split(' ')
                x_initials = [float(i) for i in x_initials]

            initiateGUI(getRow() + 1)

            if method == "Gaussian-seidel" or method == "All Methods":
                j = 0
                for i in x_initials:
                    init_guesses[j].delete(0, len(init_guesses[j].get()))
                    init_guesses[j].insert(0, i)
                    j = j + 1

            j = 0
            for i in range(2, length):
                entries[j].insert(0, fileContent[i])
                j = j + 1
        file.close()
    except:
        root_label.config(text="Error in file")
        root_label.grid(row=2, column=5)
        return


if __name__ == '__main__':
    result = []
    eachIter = []
    eachEa = []
    iterVar = 0
    window = Tk()
    fig, plot1 = subplots(figsize=(4, 4))

    color = "black"
    fg_color = "white"
    img = PhotoImage(file="background.png")
    label = Label(
        window,
        image=img
    )
    label.place(x=0, y=0)
    # setting the title
    window.title('Plotting in Tkinter')
    window.geometry("1350x700")
    window.resizable(0, 0)

    row = 0

    tkinter.Label(window, text="Precision:", bg=color, foreground=fg_color).grid(row=row, column=0)
    prec = tkinter.Entry(window, validate='key', vcmd=(window.register(validate_float), '%P'))
    prec.insert(0, "0.00001")
    prec.grid(row=row, column=1)
    row = row + 1

    tkinter.Label(window, text="Maximum Number of iterations:", bg=color, foreground=fg_color).grid(row=row, column=0)
    iters = tkinter.Entry(window, validate='key', vcmd=(window.register(validate_int), '%P'))
    iters.insert(0, "50")
    iters.grid(row=row, column=1)
    iters.setvar(value=1)
    row = row + 1

    label_eqNum = tkinter.Label(window, text="Number of equations: ", bg=color, foreground=fg_color)
    label_eqNum.grid(row=row, column=0)
    entry_eqNum = tkinter.Entry(window, vcmd=(window.register(validate_equation), '%P'))
    entry_eqNum.grid(row=row, column=1)
    row = row + 1

    global lb
    lb = []
    global init_guesses
    init_guesses = []
    global x_initials
    x_initials = []

    labels = []
    entries = []

    methodnames = tkinter.ttk.Combobox(window, width=19, textvariable=4, state="readonly")
    #
    # Adding combobox drop down list
    methodnames['values'] = (' Gauss Elimination',
                             ' LU Decomposition',
                             ' Gauss Jordan',
                             ' Gauss Seidel',
                             ' All Methods'
                             )

    label_method = tkinter.Label(window, text="Method: ", bg=color, foreground=fg_color)
    label_method.grid(row=getRow(), column=0)

    methodnames.grid(column=1, row=row)
    methodnames.current(0)
    methodnames.bind("<<ComboboxSelected>>", get_combo_box)

    row = row + 1

    Button(master=window,
           command=lambda: initiateGUI(row + 1),
           height=2,
           width=10,
           text="Enter",
           bg="cyan").grid(row=row, column=0, pady=10)

    row = row + 1

    browse = Button(master=window,
                    command=lambda: readFile(),
                    height=2,
                    width=10,
                    text="Browse",
                    bg="orange")
    browse.grid(row=2, column=3, padx=10)

    plot_button = tkinter.Button(master=window,
                                 command=lambda: plot(prec.get()),
                                 height=2,
                                 width=10,
                                 text="Plot",
                                 bg="sky blue")
    disable(plot_button)
    plot_button.grid(row=getRow() - 2, column=1, padx=30)

    root_str = ""
    root_label = tkinter.Label(window, text=root_str, bg=color, foreground=fg_color)

    # run the gui
    window.mainloop()
