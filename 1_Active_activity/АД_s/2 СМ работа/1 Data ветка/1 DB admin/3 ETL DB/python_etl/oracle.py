import oracledb

oracle_conn = oracledb.connect('excel_user/Excel123@localhost:1521/FREEPDB1')
oracle_cursor = oracle_conn.cursor()

oracle_cursor.execute('SELECT table_name FROM user_tables')
tables = oracle_cursor.fetchall()
for table in tables:
    print(table[0])