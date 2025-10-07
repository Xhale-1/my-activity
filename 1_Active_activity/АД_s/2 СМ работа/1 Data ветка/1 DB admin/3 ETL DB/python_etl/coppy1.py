import oracledb
import getpass
import os
import datetime


oracledb.init_oracle_client(lib_dir=r"D:\1a_rudms_programs\oracle\Oracle Instance Client (OIC64)\instantclient-basic-windows.x64-23.8.0.25.04 (1)\instantclient_23_8")
# --- НАСТРОЙКИ ---
# Раскомментируйте и укажите путь, если Oracle Instant Client не в системном PATH
# oracledb.init_oracle_client(lib_dir=r"C:\path\to\your\instantclient")

# Имя пользователя (схемы), структуру которой мы хотим скопировать.
DB_USER = 'E3_ADMIN'

# Строка подключения (DSN). Замените, если ваша отличается.
DB_DSN = 'pme3app1:1521/E3P2'

# Имя файла, в который будет сохранен результат.
OUTPUT_SQL_FILE = 'schema_recreation_with_pk.sql'

# --- КОНЕЦ НАСТРОЕК ---


def format_oracle_type(col_info):
    """
    Форматирует тип данных столбца в правильную строку для SQL.
    Включает обработку RAW -> BLOB.
    """
    data_type = col_info['data_type']
    char_length = col_info['char_length']
    data_precision = col_info['data_precision']
    data_scale = col_info['data_scale']

    # --- НОВОЕ: Преобразуем устаревшие типы в BLOB ---
    if data_type in ('RAW', 'LONG RAW'):
        return "BLOB"

    if data_type in ('VARCHAR2', 'CHAR', 'NVARCHAR2'):
        return f"{data_type}({char_length} CHAR)"
    
    if data_type == 'NUMBER':
        if data_precision is not None and data_scale is not None:
            if data_scale > 0:
                return f"NUMBER({data_precision}, {data_scale})"
            elif data_scale == 0:
                return f"NUMBER({data_precision}, 0)"
        return "NUMBER"

    if data_type == 'FLOAT':
        if data_precision is not None:
            return f"FLOAT({data_precision})"
        return "FLOAT"

    # Обрабатываем типы с уже включенными параметрами, например TIMESTAMP(6)
    if '(' in data_type:
         return data_type

    return data_type


def get_primary_key_clause(cursor, table_name):
    """
    Получает определение PRIMARY KEY для таблицы.
    """
    pk_columns = []
    cursor.execute("""
        SELECT ucc.column_name
        FROM user_constraints uc
        JOIN user_cons_columns ucc ON uc.constraint_name = ucc.constraint_name
        WHERE uc.table_name = :t_name
          AND uc.constraint_type = 'P' -- 'P' for Primary Key
        ORDER BY ucc.position
    """, t_name=table_name)
    
    for row in cursor.fetchall():
        pk_columns.append(f'"{row[0]}"')
    
    if pk_columns:
        return f'PRIMARY KEY ({", ".join(pk_columns)}) ENABLE'
    
    return None

def main():
    """
    Главная функция скрипта.
    """
    try:
        # Для простоты захардкодим пароль, как в вашем примере.
        # В реальных системах лучше использовать getpass или переменные окружения.
        db_password = "ddbadmine3"

        print(f"Подключение к {DB_DSN}...")
        
        with oracledb.connect(user=DB_USER, password=db_password, dsn=DB_DSN) as connection:
            print("Успешное подключение.")
            with connection.cursor() as cursor:
                
                cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
                table_names = [row[0] for row in cursor.fetchall()]
                
                if not table_names:
                    print("В схеме не найдено таблиц.")
                    return

                print(f"Найдено {len(table_names)} таблиц. Генерация DDL...")

                all_create_statements = []

                for table_name in table_names:
                    print(f"  - Обработка таблицы: {table_name}")
                    
                    cursor.execute("""
                        SELECT column_name, data_type, char_length, data_precision, 
                               data_scale, nullable
                        FROM user_tab_columns
                        WHERE table_name = :t_name
                        ORDER BY column_id
                    """, t_name=table_name)

                    columns_data = cursor.fetchall()
                    
                    column_definitions = []
                    for col in columns_data:
                        col_info = dict(zip(['name', 'data_type', 'char_length', 'data_precision', 'data_scale', 'nullable'], col))
                        
                        formatted_type = format_oracle_type(col_info)
                        
                        # --- ИЗМЕНЕНИЕ: Добавляем ENABLE к NOT NULL ---
                        not_null_clause = " NOT NULL ENABLE" if col_info['nullable'] == 'N' else ""
                        
                        column_definitions.append(f'    "{col_info["name"]}" {formatted_type}{not_null_clause}')

                    # --- НОВОЕ: Получаем информацию о первичном ключе ---
                    pk_clause = get_primary_key_clause(cursor, table_name)
                    if pk_clause:
                        column_definitions.append(f"    {pk_clause}")

                    # Собираем полный CREATE TABLE запрос
                    # --- ИЗМЕНЕНИЕ: Добавляем имя схемы ---
                    create_statement = f'CREATE TABLE "{DB_USER}"."{table_name}" (\n'
                    create_statement += ",\n".join(column_definitions)
                    create_statement += '\n);\n'
                    
                    all_create_statements.append(create_statement)

                # Запись в файл
                print(f"\nЗапись DDL в файл '{OUTPUT_SQL_FILE}'...")
                with open(OUTPUT_SQL_FILE, 'w', encoding='utf-8') as f:
                    f.write(f"-- Скрипт для воссоздания схемы {DB_USER}\n")
                    f.write(f"-- Сгенерировано: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    for statement in all_create_statements:
                        f.write(statement + '\n')
                        
                print(f"Готово! Файл '{os.path.abspath(OUTPUT_SQL_FILE)}' успешно создан.")

    except oracledb.DatabaseError as e:
        error, = e.args
        print(f"Ошибка Oracle: {error.code} - {error.message}")
        if "DPI-1047" in error.message:
            print("\nПодсказка: Эта ошибка часто означает, что Python не может найти Oracle Instant Client.")
            print("Убедитесь, что путь в oracledb.init_oracle_client() указан верно или Instant Client находится в системной переменной PATH.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")


if __name__ == "__main__":
    main()