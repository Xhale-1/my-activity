from datetime import datetime
import pyodbc 
import oracledb

# Подключение к Access
access_conn = pyodbc.connect(
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=D:\Zuken\Examples\cdb\own_components_and_symbols_and_configuration.mdb;'
)
access_cursor = access_conn.cursor()

# Подключение к Oracle
oracle_conn = oracledb.connect('E3_ADMIN/ddbadmine3@localhost:1521/FREEPDB1')
oracle_cursor = oracle_conn.cursor()

access_cursor.tables(tableType='TABLE')  # Получает только пользовательские таблицы
tables = [table.table_name for table in access_cursor.fetchall()]

def infer_oracle_type(value):
    if isinstance(value, int):
        return "NUMBER"
    elif isinstance(value, float):
        return "FLOAT"
    elif isinstance(value, (bytes, bytearray, memoryview)):
        return "BLOB"
    elif isinstance(value, str):
        return "VARCHAR2(2000)"
    elif isinstance(value, datetime):
        return "TIMESTAMP"
    else:
        return "VARCHAR2(2000)" # по умолчанию

for table in tables:
    print(table)

    access_cursor.execute(f'SELECT COUNT(*) FROM "{table}";')
    table_row_cnt = access_cursor.fetchone()
    access_cursor.execute(f'SELECT * FROM "{table}" WHERE 1=0')  # Выполняем пустой запрос
    column_count = len(access_cursor.description)

    access_cursor.execute(f'SELECT TOP 2 * FROM "{table}"')
    sample_row = access_cursor.fetchone()
    print(sample_row)

    columns = [desc[0] for desc in access_cursor.description]
    print(columns)

    if sample_row is None:
        column_types = ["VARCHAR2(2000)" for _ in columns]
    else:
        column_types = []
        for col in columns:
            access_cursor.execute(f'''
                SELECT TOP 1 [{col}] 
                FROM "{table}" 
                WHERE [{col}] IS NOT NULL
                    ''')
    
            val = access_cursor.fetchone()
            column_types.append(infer_oracle_type(val[0]) if val else "VARCHAR2(2000)")

    print(column_types)
    
    # Генерация CREATE TABLE
    create_table_sql = f"""
    CREATE TABLE "{table}" (
        {', '.join([f'"{col}" {col_type}' for col, col_type in zip(columns, column_types)])}
    )
    """

    # Проверяем, есть ли уже таблица
    oracle_cursor.execute(f"""
        SELECT table_name FROM user_tables WHERE table_name = '{table}'
    """)
    if not oracle_cursor.fetchone():
        oracle_cursor.execute(create_table_sql)
        print("Таблица создана!")

    # Вставка всех строк
    access_cursor.execute(f'SELECT * FROM "{table}"')
    rows = access_cursor.fetchall()
    placeholders = ", ".join([f":{i+1}" for i in range(len(columns))])
    insert_sql = f'INSERT INTO "{table}" VALUES ({placeholders})'

    print(placeholders)
    print('_________________')

    oracle_cursor.execute(f'SELECT * FROM "{table}" WHERE ROWNUM = 0')
    oracle_columns = [desc[0] for desc in oracle_cursor.description]
    print("Столбцы в Oracle таблице:", oracle_columns)

    print('_________________')

    # Подготовка BLOB переменной
    blob_var = oracle_cursor.var(oracledb.BLOB)

    for row in rows:
        row_data = []
        for value in row:
            if value is None:
                row_data.append(None)
            elif isinstance(value, (bytes, bytearray, memoryview)):
                # Для BLOB-данных используем переменную
                blob_var.setvalue(0, bytes(value))
                row_data.append(blob_var)
            else:
                row_data.append(value)
        
        try:
            oracle_cursor.execute(insert_sql, row_data)
        except Exception as e:
            print(f"Ошибка при вставке строки: {e}")
            print(f"SQL: {insert_sql}")
            print(f"Данные: {row_data}")
            raise

    oracle_conn.commit()
    print("Данные перенесены!")

# Закрытие соединений
access_conn.close()
oracle_conn.close()