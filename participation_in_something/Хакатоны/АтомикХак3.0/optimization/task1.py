import pulp
import pandas as pd

# --- данные ---
P_nom_DGU = 30
P_min_DGU = 0.5 * P_nom_DGU
cost_DGU = 5
cost_grid_0_6 = 9
cost_grid_7_23 = 11

data = {
    'time': [f'{h:02d}:00' for h in range(24)],
    'solar_pu': [0.6, 0.55, 0.5, 0.45, 0.4, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    'load_pu': [0.5, 0.45, 0.4, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95,
                1.0, 0.95, 0.9, 0.85, 0.9, 1.0, 0.95, 0.8, 0.7, 0.6, 0.55, 0.5]
}

P_nom_S = 15
P_nag = 60
df = pd.DataFrame(data)
df['P_solar'] = df['solar_pu'] * P_nom_S
df['P_load'] = df['load_pu'] * P_nag

# --- модель ЛП ---
model = pulp.LpProblem("Energy_Optimization", pulp.LpMinimize)

# Переменные
P_DGU = pulp.LpVariable.dicts("P_DGU", range(24), lowBound=0, upBound=P_nom_DGU)
P_grid = pulp.LpVariable.dicts("P_grid", range(24), lowBound=0)

# Целевая функция
model += pulp.lpSum([
    cost_DGU * P_DGU[t] +
    ((cost_grid_0_6 if 0 <= t <= 6 else cost_grid_7_23) * P_grid[t])
    for t in range(24)
])

# Ограничения
for t in range(24):
    P_load = df.loc[t, 'P_load']
    P_solar = df.loc[t, 'P_solar']
    model += P_DGU[t] + P_grid[t] + P_solar == P_load

    # нижний предел ДГУ, если включен
    # здесь без бинарных переменных, так что просто ограничиваем минимальное значение
    model += P_DGU[t] >= P_min_DGU

# Решение
model.solve(pulp.PULP_CBC_CMD(msg=False))


# --- результаты ---
df['P_DGU'] = [pulp.value(P_DGU[t]) for t in range(24)]
df['P_grid'] = [pulp.value(P_grid[t]) for t in range(24)]
df['total_cost'] = cost_DGU * df['P_DGU'] + [
    (cost_grid_0_6 if 0 <= t <= 6 else cost_grid_7_23) * df['P_grid'][t]
    for t in range(24)
]

print(df)
print(df['P_DGU'].sum()*cost_DGU + df['P_grid'].iloc[:7].sum()*cost_grid_0_6 + df['P_grid'].iloc[7:].sum()*cost_grid_7_23)

df_answer = df[['time','P_DGU','P_grid']]
df_answer.to_csv("./optimize_task1.csv", sep=';', decimal='.', index=False)
# print(df[['time', 'P_load', 'P_solar', 'P_DGU', 'P_grid', 'total_cost']])
# print("Total cost =", df['total_cost'].sum())
