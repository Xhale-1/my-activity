import pyodbc
import oracledb

# --- Настройки ---

# Установите True, если хотите, чтобы скрипт удалял таблицы в Oracle перед их созданием
# Установите False, если хотите пропускать создание уже существующих таблиц
DROP_EXISTING_TABLES = True 

# --- Подключения к базам данных ---

try:
    # Подключение к Access
    # Убедитесь, что у вас установлен 64-битный драйвер Microsoft Access,
    # если вы используете 64-битную версию Python.
    access_conn = pyodbc.connect(
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=D:\1a_rudms_work\zadaniya\10 e3_64 connect db\BD.accdb;'
    )
    access_cursor = access_conn.cursor()

    # # Подключение к Oracle
    # oracle_conn = oracledb.connect('E3_ADMIN/ddbadmine3@localhost:1522/FREEPDB1')
    # oracle_cursor = oracle_conn.cursor()

    print("Успешно подключено к Access и Oracle.")

    def map_access_type_to_oracle(col_desc):
        """
        Преобразует метаданные столбца из pyodbc в тип данных Oracle.
        col_desc - это кортеж из cursor.description.
        (name, type_code, display_size, internal_size, precision, scale, null_ok)
        """
        type_code = col_desc[1]
        display_size = col_desc[2] or 2000 # Размер по умолчанию
        precision = col_desc[4]
        scale = col_desc[5]

        # Сопоставление типов
        if type_code == pyodbc.SQL_CHAR or type_code == pyodbc.SQL_VARCHAR:
            # Ограничиваем максимальный размер для VARCHAR2
            size = min(display_size, 4000)
            return f"VARCHAR2({size} CHAR)"
        elif type_code == pyodbc.SQL_LONGVARCHAR or type_code == pyodbc.SQL_WLONGVARCHAR:
            return "CLOB"
        elif type_code in (pyodbc.SQL_INTEGER, pyodbc.SQL_SMALLINT):
            return "NUMBER(10)"
        elif type_code == pyodbc.SQL_BIGINT:
            return "NUMBER(19)"
        elif type_code == pyodbc.SQL_BIT: # Тип "Да/Нет" в Access
            return "NUMBER(1)"
        elif type_code in (pyodbc.SQL_DECIMAL, pyodbc.SQL_NUMERIC):
            if precision > 0:
                return f"NUMBER({precision}, {scale or 0})"
            return "NUMBER" # Общий числовой тип
        elif type_code in (pyodbc.SQL_DOUBLE, pyodbc.SQL_FLOAT, pyodbc.SQL_REAL):
            return "FLOAT"
        elif type_code in (pyodbc.SQL_TYPE_DATE, pyodbc.SQL_TYPE_TIME, pyodbc.SQL_TYPE_TIMESTAMP):
            return "TIMESTAMP"
        elif type_code in (pyodbc.SQL_BINARY, pyodbc.SQL_VARBINARY, pyodbc.SQL_LONGVARBINARY):
            # OLE Object в Access обычно попадает сюда
            return "BLOB"
        else:
            # Тип по умолчанию для неизвестных типов
            return "VARCHAR2(2000 CHAR)"

    # Получаем список всех пользовательских таблиц из Access
    access_cursor.tables(tableType='TABLE')
    tables = [table.table_name for table in access_cursor.fetchall() if not table.table_name.startswith('~')]

    print(f"\nНайдено таблиц в Access: {len(tables)}")
    print("----------------------------------------")

    for table_name in tables:
        print(f"Обработка таблицы: '{table_name}'")

        try:
            # Проверяем, существует ли таблица в Oracle
            # Имена таблиц в Oracle по умолчанию хранятся в верхнем регистре
            # oracle_cursor.execute(f"SELECT table_name FROM user_tables WHERE table_name = '{table_name.upper()}'")
            # table_exists = oracle_cursor.fetchone()

            # if table_exists and DROP_EXISTING_TABLES:
            #     print(f"  - Таблица '{table_name.upper()}' уже существует. Удаляем...")
            #     oracle_cursor.execute(f'DROP TABLE "{table_name.upper()}" CASCADE CONSTRAINTS')
            #     print(f"  - Таблица '{table_name.upper()}' удалена.")
            # elif table_exists:
            #     print(f"  - Таблица '{table_name.upper()}' уже существует. Пропускаем.")
            #     continue

            # Получаем метаданные столбцов, не извлекая данные
            access_cursor.execute(f'SELECT * FROM "{table_name}" WHERE 1=0')
            
            columns_metadata = access_cursor.description
            if not columns_metadata:
                print(f"  - Не удалось получить метаданные для таблицы '{table_name}'. Пропускаем.")
                continue
            
            column_definitions = []
            for col_meta in columns_metadata:
                col_name = col_meta[0]
                oracle_type = map_access_type_to_oracle(col_meta)
                column_definitions.append(f'"{col_name}" {oracle_type}')

            # Генерируем SQL-запрос для создания таблицы
            create_table_sql = f"""
            CREATE TABLE "{table_name.upper()}" (
                {', '.join(column_definitions)}
            )
            """
            
            print("  - Генерирую CREATE TABLE скрипт...")
            # print(create_table_sql) # Раскомментируйте для отладки

            # Выполняем создание таблицы в Oracle
            # oracle_cursor.execute(create_table_sql)
            # print(f"  - Таблица '{table_name.upper()}' успешно создана в Oracle.")

        except Exception as e:
            print(f"  - ОШИБКА при обработке таблицы '{table_name}': {e}")
            #oracle_conn.rollback() # Откатываем транзакцию в случае ошибки с одной таблицей
    
    # DDL (CREATE, DROP) в Oracle часто имеют автокоммит, но явный commit не повредит
    #oracle_conn.commit()
    print("----------------------------------------")
    print("Создание схемы завершено.")

finally:
    # Гарантированное закрытие соединений
    if 'access_conn' in locals() and access_conn:
        access_conn.close()
        print("\nСоединение с Access закрыто.")
    if 'oracle_conn' in locals() and oracle_conn:
        oracle_conn.close()
        print("Соединение с Oracle закрыто.")