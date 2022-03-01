import pyrez
import db

devId = 3036
authKey = 'E00ABB875BD945E1ACC8119F12BE27AD'
dbConn = db.DataBase()


def process_matches():
    with pyrez.SmiteAPI(devId, authKey) as smite:
        minutes = [',00', ',10', ',20', ',30', ',40', ',50']
        for day in range(21, 22):
            for hour in range(24):
                for minute in minutes:
                    ids = smite.getMatchIds(451, '2022-02-0' + str(day), str(hour) + minute)
                    print('2022-02-' + str(day) + ' ' + str(hour) + minute)
                    for i in ids:
                        match = smite.getMatch(i.matchId)

                        # test same model for lower level players
                        # Check that we only take high level matches and avoid disconnects also consider only matches longer than 10 min
                        num_experienced = len([player for player in match if player.accountLevel > 30])
                        if num_experienced == 10 and match[0].matchDuration > 600:
                            team_1 = tuple(map(lambda player: int(player.GodId), filter(lambda player: player.TaskForce == 1, match)))
                            team_2 = tuple(map(lambda player: int(player.GodId), filter(lambda player: player.TaskForce == 2, match)))
                            bans = (int(match[0].Ban1Id), int(match[0].Ban2Id), int(match[0].Ban3Id), int(match[0].Ban4Id), int(match[0].Ban5Id), int(match[0].Ban6Id), int(match[0].Ban7Id), int(match[0].Ban8Id), int(match[0].Ban9Id), int(match[0].Ban10Id))
                            win = (int(match[0].Winning_TaskForce),)
                            print((int(i.matchId),) + team_1 + team_2 + bans + win)
                            dbConn.insert("INSERT INTO Match (id,"
                                          "t1_god1, t1_god2, t1_god3, t1_god4, t1_god5, "
                                          "t2_god1, t2_god2, t2_god3, t2_god4, t2_god5, "
                                          "ban1, ban2, ban3, ban4, ban5, ban6, ban7, ban8, ban9, ban10, win)"
                                          "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                          (int(i.matchId),) + team_1 + team_2 + bans + win)


def insert_gods():
    with pyrez.SmiteAPI(devId, authKey) as smite:
        gods = [{'name': god.Name, 'id': int(god.godId)} for god in smite.getGods()]
        for god in gods:
            dbConn.insert("INSERT INTO public.god (id, name) VALUES (%s, %s)",
                          (god['id'], god['name']))


dbConn.create_tables()
#dbConn.empty_database()
#insert_gods()
process_matches()
