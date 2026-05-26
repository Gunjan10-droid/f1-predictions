import requests
import pandas as pd
import time

def get_race_results(start_year=2000, end_year=2026):
    all_races = []
    
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year}...")
        offset = 0
        
        while True:
            url = f"https://api.jolpi.ca/ergast/f1/{year}/results.json?limit=30&offset={offset}"
            
            while True:
                try:
                    r = requests.get(url, timeout=15)
                    
                    if r.status_code == 429:
                        print(f"  Rate limited, waiting 10s...")
                        time.sleep(10)
                        continue
                    
                    if r.status_code != 200:
                        print(f"  Bad status {r.status_code}, skipping...")
                        break
                    
                    if not r.text.strip():
                        print(f"  Empty response, skipping...")
                        break
                    
                    data = r.json()["MRData"]
                    races = data["RaceTable"]["Races"]
                    total = int(data["total"])
                    
                    if not races:
                        break
                        
                    for race in races:
                        for result in race["Results"]:
                            all_races.append({
                                "year":        race["season"],
                                "round":       race["round"],
                                "circuit_id":  race["Circuit"]["circuitId"],
                                "race_name":   race["raceName"],
                                "date":        race["date"],
                                "driver_id":   result["Driver"]["driverId"],
                                "driver_name": result["Driver"]["givenName"] + " " + result["Driver"]["familyName"],
                                "constructor": result["Constructor"]["constructorId"],
                                "grid":        int(result["grid"]),
                                "position":    result["positionText"],
                                "points":      float(result["points"]),
                                "status":      result["status"],
                                "laps":        int(result["laps"]),
                            })
                    
                    offset += 30
                    if offset >= total:
                        break
                    
                    time.sleep(2)
                    break
                    
                except Exception as e:
                    print(f"  Error: {e}, retrying in 5s...")
                    time.sleep(5)
                    continue
            
            else:
                break
            
            if not races:
                break
        
        time.sleep(2)
    
    df = pd.DataFrame(all_races)
    df.to_csv("data/raw/race_results.csv", index=False)
    print(f"\nDone! {len(df)} rows saved to data/raw/race_results.csv")
    return df


if __name__ == "__main__":
    df = get_race_results()
    print(df.head(10))
