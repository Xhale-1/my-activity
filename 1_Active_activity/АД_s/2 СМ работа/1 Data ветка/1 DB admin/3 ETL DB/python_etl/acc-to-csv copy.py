import pyodbc
import oracledb
import csv
import os
from pathlib import Path

# --- НАСТРОЙКИ ---
ACCESS_DB_PATH = r"D:\Zuken\Examples\cdb\own_components_and_symbols_and_configuration.mdb"
OUTPUT_DIR = Path(r".\python_etl\csv")

# Access ODBC
DRIVER_STRING = r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"

# Oracle
DB_USER = "E3_ADMIN"
DB_PASS = "ddbadmine3"
DB_DSN  = "10.11.77.245:1521/FREEPDB1"
# --- КОНЕЦ НАСТРОЕК ---


def get_oracle_columns(table_name, cursor):
    """Берём список колонок таблицы в Oracle в правильном порядке"""
    sql = """
        SELECT COLUMN_NAME
        FROM ALL_TAB_COLUMNS
        WHERE OWNER = :owner
          AND TABLE_NAME = :table_name
        ORDER BY COLUMN_ID
    """
    cursor.execute(sql, owner=DB_USER.upper(), table_name=table_name)
    return [row[0] for row in cursor.fetchall()]


def export_access_to_csv():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    conn_str = f"{DRIVER_STRING}DBQ={ACCESS_DB_PATH};"

    print(f"Подключаюсь к Access: {ACCESS_DB_PATH}")
    print(f"Подключаюсь к Oracle: {DB_DSN}")

    try:
        # --- Oracle connect
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as ora_conn:
            ora_cursor = ora_conn.cursor()

            # --- Access connect
            with pyodbc.connect(conn_str) as access_conn:
                access_cursor = access_conn.cursor()

                # Получаем список всех таблиц в Access
                table_names = []
                for table_info in access_cursor.tables(tableType="TABLE"):
                    tname = table_info.table_name
                    if not tname.startswith("MSys"):  # системные исключаем
                        table_names.append(tname)

                print(f"Найдено {len(table_names)} таблиц: {', '.join(table_names)}")

                for table_name in table_names:
                    print(f"\nЭкспортируем {table_name}...")

                    # Получаем список колонок в Oracle
                    oracle_cols = get_oracle_columns(table_name, ora_cursor)
                    if not oracle_cols:
                        print(f" ⚠️ В Oracle таблицы {table_name} нет — пропуск")
                        continue

                    # Читаем все данные из Access
                    sql_query = f"SELECT * FROM [{table_name}]"
                    access_cursor.execute(sql_query)

                    access_headers = [col[0] for col in access_cursor.description]
                    rows = [dict(zip(access_headers, row)) for row in access_cursor.fetchall()]

                    # Формируем CSV по Oracle‑порядку колонок
                    csv_path = OUTPUT_DIR / f"{table_name}.csv"
                    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(oracle_cols)  # заголовки — как в Oracle

                        # Стало (с обработкой bytes, True и False):
                        for row in rows:
                            csv_row = []
                            for col in oracle_cols:
                                value = row.get(col, "")
                                
                                # Сначала проверяем, не байтовый ли это объект
                                if isinstance(value, bytes):
                                    csv_row.append(value.hex()) # Кодируем в HEX-строку
                                elif value is True:
                                    csv_row.append(-1)
                                elif value is False:
                                    csv_row.append(0)
                                else:
                                    csv_row.append(value)
                                    
                            writer.writerow(csv_row)

                    print(f" ✅ Сохранено: {csv_path} ({len(rows)} строк)")

    except Exception as e:
        print("❌ Ошибка:", e)


if __name__ == "__main__":
    export_access_to_csv()