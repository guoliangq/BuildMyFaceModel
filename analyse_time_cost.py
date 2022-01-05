

def read_file(filename):
    print(f"in {filename}: ")
    res = []
    file  = open(filename)
    lines = file.readlines()
    file.close()
    for line in lines:
        res.append(float(line.split(" ")[1]))
    return res

def compute(list):
    max = -1.
    min = 100.
    count = 0
    sum = 0.
    for num in list:
        if num > max :
            max = num
        if num < min:
            min = num
        count += 1
        sum += num
    avg = sum/count
    print(f"max:{max},min:{min},avg:{avg}")

if __name__ == "__main__":
    compute(read_file("all_cost.txt"))
    compute(read_file("yolo_cost.txt"))
    compute(read_file("face_cost.txt"))
    compute(read_file("spoof_cost.txt"))