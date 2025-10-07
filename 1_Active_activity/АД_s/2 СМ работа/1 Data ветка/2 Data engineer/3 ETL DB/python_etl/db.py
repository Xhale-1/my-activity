import pyodbc


db_path = r'D:\1a_rudms_work\zadaniya\10 e3_64 connect db\БД_ДСАЭМ_КОСААСУТП.mdb'  
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=' + db_path + ';'
    # r'Exclusive=1;'  # ← Добавьте это для монопольного доступа
)

try:
    # Подключение к БД
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # cursor.execute('SELECT COUNT(*) FROM "ComponentData"')
    # print(f"Записей в SomeTable: {cursor.fetchone()[0]}")


    tables = cursor.tables(tableType='TABLE')  # Тип 'TABLE' = обычные таблицы
    print("Список таблиц в базе:")
    for table in tables:
        print(table.table_name)
    
    # Закрытие соединения
    cursor.close()
    conn.close()
    print("База данных работоспособна!")
    
except Exception as e:
    print(f"Ошибка подключения или выполнения запроса: {e}")