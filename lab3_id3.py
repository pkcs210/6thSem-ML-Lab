import math

def find_entropy(x, y):
    if x != 0 and y != 0:
        entropy = -1 * (x * math.log2(x) + y * math.log2(y))
        return entropy
    if x == 1:
        return 1
    if y == 1:
        return 0
    return 0
    
def find_max_gain(data, rows, columns):
    max_gain = 0
    retidx = -1
    entropy_ans = find_entropy(data, rows)
    if entropy_ans == 0:
        return max_gain, retidx, entropy_ans

    for j in columns:
        mydict = {}
        ddd = 1
        for i in rows:
            key = data[i][j]
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] += 1

        gain = entropy_ans
        for key in mydict:
            yes = 0
            no = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 1:  # Assuming last column is the label
                        yes += 1
                    else:
                        no += 1
            x = yes / (yes + no)
            y = no / (yes + no)
            if x != 0 and y != 0:
                gain += (mydict[key] * (x * math.log2(x) + y * math.log2(y))) / 14  # 14 is total samples

        if gain > max_gain:
            max_gain = gain
            retidx = j

    return max_gain, retidx, entropy_ans
    
def build_tree(X, rows, columns):
    max_gain, best_col, entropy = find_max_gain(X, rows, columns)
    if best_col == -1 or entropy == 0:
        return

