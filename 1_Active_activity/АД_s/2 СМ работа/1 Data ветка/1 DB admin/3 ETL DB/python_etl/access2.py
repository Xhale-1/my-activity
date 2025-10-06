import win32com.client
import os

def mdb_to_accdb(mdb_path, accdb_path=None):
    """Конвертирует MDB в ACCDB через COM-объект Access"""
    if not os.path.exists(mdb_path):
        raise FileNotFoundError(f"Файл не найден: {mdb_path}")
    
    if accdb_path is None:
        accdb_path = os.path.splitext(mdb_path)[0] + ".accdb"

    try:
        access = win32com.client.Dispatch("Access.Application")
        
        # Константа для формата ACCDB (Access 2007+)
        acFileFormatAccess12 = 12  

        # Правильный вызов ConvertAccessProject
        access.ConvertAccessProject(
            mdb_path,          # Исходный файл .mdb
            accdb_path,         # Целевой файл .accdb
            acFileFormatAccess12  # Формат ACCDB
        )
        
        access.Quit()
        print(f"Конвертация завершена: {accdb_path}")
        return accdb_path
    except Exception as e:
        raise Exception(f"Ошибка COM: {e}")

# Установка: pip install pywin32
mdb_to_accdb(r"D:\1a_rudms_work\zadaniya\11_recover_DB\БД_ДСАЭМ_КОСААСУТП LOST.mdb", r"D:\1a_rudms_work\zadaniya\11_recover_DB\БД_ДСАЭМ_КОСААСУТП LOST2.accdb")