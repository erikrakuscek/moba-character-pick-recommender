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

    def insert(self, query, params):
        cur = self.conn.cursor()
        cur.execute(query, params)
        cur.close()
        return True

    def empty_database(self):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM match")
        cur.close()
        return True

    def create_tables(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS Match (" +
                    "id SERIAL PRIMARY KEY," +
                    "t1_god1 INT," +
                    "t1_god2 INT," +
                    "t1_god3 INT," +
                    "t1_god4 INT," +
                    "t1_god5 INT," +
                    "t2_god1 INT," +
                    "t2_god2 INT," +
                    "t2_god3 INT," +
                    "t2_god4 INT," +
                    "t2_god5 INT," +
                    "ban1 INT," +
                    "ban2 INT," +
                    "ban3 INT," +
                    "ban4 INT," +
                    "ban5 INT," +
                    "ban6 INT," +
                    "ban7 INT," +
                    "ban8 INT," +
                    "ban9 INT," +
                    "ban10 INT," +
                    "win INT" +
                    ");")
        cur.execute("CREATE TABLE IF NOT EXISTS Player (" +
                    "id SERIAL PRIMARY KEY," +
                    "match_id INT," +
                    "active1 INT," +
                    "active2 INT," +
                    "item1 INT," +
                    "item2 INT," +
                    "item3 INT," +
                    "item4 INT," +
                    "item5 INT," +
                    "item6 INT);")
        cur.execute("CREATE TABLE IF NOT EXISTS God (" +
                    "id INT PRIMARY KEY," +
                    "name TEXT" +
                    ");")
        cur.close()
        return True

    def select(self, query, params):
        cur = self.conn.cursor()
        cur.execute(query, params)
        value = cur.fetchall()
        cur.close()
        return value
