import pyodbc
import csv
import os

# --- НАСТРОЙКИ ---

# 1. Укажите полный путь к вашему файлу базы данных Access
ACCESS_DB_PATH = r'D:\Zuken\Examples\cdb\own_components_and_symbols_and_configuration.mdb'

# 2. Укажите папку, куда будут сохраняться CSV-файлы
OUTPUT_DIR = r'.\access_export'

# 3. Строка подключения (обычно не требует изменений)
# Убедитесь, что у вас установлен соответствующий драйвер (32-bit или 64-bit)
DRIVER_STRING = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'

# --- КОНЕЦ НАСТРОЕК ---


def export_access_to_csv():
    """
    Подключается к базе данных MS Access, находит все таблицы и экспортирует
    каждую в отдельный CSV-файл в указанную директорию.
    """
    # Проверяем, существует ли папка для вывода, и создаем ее, если нет
    if not os.path.exists(OUTPUT_DIR):
        print(f"Создаю директорию для вывода: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    # Формируем полную строку подключения
    conn_str = f'{DRIVER_STRING}DBQ={ACCESS_DB_PATH};'

    print(f"Подключаюсь к базе данных: {ACCESS_DB_PATH}...")
    
    try:
        # Устанавливаем соединение с базой данных
        with pyodbc.connect(conn_str) as access_conn:
            access_cursor = access_conn.cursor()

            # 1. Получаем список всех пользовательских таблиц в базе данных
            # Мы фильтруем таблицы, чтобы исключить системные (которые обычно начинаются с 'MSys')
            table_names = []
            for table_info in access_cursor.tables(tableType='TABLE'):
                table_name = table_info.table_name
                if not table_name.startswith('MSys'):
                    table_names.append(table_name)
            
            print(f"Найдено таблиц для экспорта: {len(table_names)}")
            print(', '.join(table_names))
            print("-" * 30)

            # 2. Проходим по каждой таблице и экспортируем ее данные
            for table_name in table_names:
                print(f"Экспортирую таблицу '{table_name}'...")

                # Формируем безопасный SQL-запрос для извлечения всех данных
                # Квадратные скобки нужны на случай, если в имени таблицы есть пробелы или спецсимволы
                sql_query = f"SELECT * FROM [{table_name}]"
                
                # Выполняем запрос
                access_cursor.execute(sql_query)
                
                # Получаем заголовки столбцов из описания курсора
                headers = [column[0] for column in access_cursor.description]
                
                # Получаем все строки данных
                rows = access_cursor.fetchall()
                
                # Формируем путь к выходному CSV-файлу
                csv_path = os.path.join(OUTPUT_DIR, f"{table_name}.csv")

                # 3. Записываем данные в CSV-файл
                # Используем кодировку utf-8 для лучшей совместимости
                # newline='' - важный параметр для корректной записи строк в CSV
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Сначала записываем строку с заголовками
                    writer.writerow(headers)
                    
                    # Затем записываем все строки данных
                    writer.writerows(rows)

                print(f" -> Сохранено в файл: {csv_path} ({len(rows)} строк)")

    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"ОШИБКА при работе с базой данных: {sqlstate}")
        print(ex)
        print("\nВозможная причина: Неправильный путь к файлу, или версия драйвера Access (32/64-bit) не совпадает с версией Python.")
    
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")

    else:
        print("-" * 30)
        print("Экспорт успешно завершен!")


# Запуск основной функции
if __name__ == "__main__":
    export_access_to_csv()