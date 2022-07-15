import psycopg2


class DataBase:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="SmiteDataNew",
            user="postgres",
            password="wuboteam"
        )
        self.conn.autocommit = True

    def select(self, query, params):
        cur = self.conn.cursor()
        cur.execute(query, params)
        value = cur.fetchall()
        cur.close()
        return value
