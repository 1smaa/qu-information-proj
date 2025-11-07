import os

n_lines=[0,0,0,0]
for entry in os.listdir("data"):
    with open(os.path.join("data",entry),mode="r") as f:
        lines=f.readlines()
    lines=lines[6:]
    n_lines[3]+=len(lines)
    for line in lines:
        ts,ch=line.split(";")
        ts=ts.strip()
        ch=int(ch.strip())
        n_lines[ch-1]+=1
print(n_lines)