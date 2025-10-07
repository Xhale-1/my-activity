import pyodbc

mdb_conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
r'DBQ=D:\1a_rudms_work\zadaniya\11_recover_DB\БД_ДСАЭМ_КОСААСУТП LOST — копия.accdb;')

mdb_conn = pyodbc.connect(mdb_conn_str)
mdb_cursor = mdb_conn.cursor()

tables = mdb_cursor.tables(tableType='TABLE')
print(tables)
table_names = [table.table_name for table in tables]
print(table_names)

#mdb_cursor.execute(f'SELECT * FROM "ComponentData"')
#rows = mdb_cursor.fetchone()
#print(rows)

# cursor.tables(tableType='TABLE')  # Получает только пользовательские таблицы
# tables = cursor.fetchall()

# for table in tables:
#     print(table.table_name)  # Выводим имена таблиц

# # Закрытие соединения
# cursor.close()
# conn.close()