import pandas as pd
import random

data = []

for i in range(500):

    battery = random.randint(20, 100)
    temp = random.randint(25, 50)
    speed = random.randint(20, 120)

    voltage = round(random.uniform(40, 50), 2)
    current = round(random.uniform(5, 30), 2)

    soh = round(100 - (100 - battery) * 0.1, 2)

    ev_range = round(battery * 2 - speed * 0.3, 2)

    data.append([
        battery,
        temp,
        speed,
        voltage,
        current,
        soh,
        ev_range
    ])

df = pd.DataFrame(data, columns=[
    "battery",
    "temp",
    "speed",
    "voltage",
    "current",
    "soh",
    "range"
])

df.to_csv("ev_live_data.csv", index=False)

print("CSV file created successfully!")
