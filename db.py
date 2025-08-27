"""
Refactored DB API client for DB (bahn.de) with dataclasses and utility functions.

This version adds an interactive flow that:
 - Asks the user for source, destination, departure datetime (optional) and train number
 - Resolves stations using the station search API (user chooses one)
 - Fetches connections for the chosen route and finds the specific connection by train number
 - Extracts the ordered stops the train visits
 - Queries prices for every forward segment the train covers (start stop -> any later stop)
 - Builds a directed acyclic graph (only forward edges) with edge weights = segment prices
 - Computes the cheapest path (no train switching) from the user's chosen origin stop to the chosen destination stop
 - Ensures the chosen path actually corresponds to segments the identified train runs

Notes / assumptions:
 - The code assumes the API returns fields similar to your sample JSON ("verbindungen" / "verbindungsAbschnitte" / "halte" / "angebotsPreis" / "verkehrsmittel.nummer" etc.).
 - Station identifiers used with the "fahrplan" endpoint are taken from the station search results -> typically an "id" string that looks like the A=...@O=...@... format.
 - Calls to the remote API may be rate-limited. We include short sleeps between many price queries to avoid hammering the API.

"""

from __future__ import annotations
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Tuple, Any
import heapq
import json
import time
import random

#######################################################################
# SET THIS TO YOUR LOCAL FOLDER PATH, MINE IS just_for_fun
local_folder = 'just_for_fun'
#######################################################################

# --------------------------- Data classes ---------------------------------

@dataclass
class StationInfo:
    raw: str
    name: Optional[str]
    id: Optional[str]
    x: Optional[int] = None
    y: Optional[int] = None


@dataclass
class Price:
    betrag: float
    waehrung: str


@dataclass
class Halt:
    id: Optional[str]
    name: Optional[str]
    extId: Optional[str]
    abfahrtsZeitpunkt: Optional[datetime]
    ankunftsZeitpunkt: Optional[datetime]
    gleis: Optional[str]
    routeIdx: Optional[int]


@dataclass
class ConnectionSection:
    idx: int
    journeyId: Optional[str]
    verkehrsmittel_name: Optional[str]
    verkehrsmittel_nummer: Optional[str]
    halte: List[Halt]
    abfahrtsZeitpunkt: Optional[datetime]
    ankunftsZeitpunkt: Optional[datetime]
    dauer_seconds: Optional[int]


@dataclass
class Connection:
    tripId: str
    sections: List[ConnectionSection]
    dauer_seconds: int
    preis: Optional[Price]
    umstiege: int


# --------------------------- Utilities -----------------------------------

def _parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt:
        return None
    try:
        return datetime.fromisoformat(dt)
    except Exception:
        try:
            # remove trailing Z if present
            return datetime.fromisoformat(dt.replace('Z', ''))
        except Exception:
            return None


def parse_halt(halt_json: Dict[str, Any]) -> Halt:
    return Halt(
        id=halt_json.get('id'),
        name=halt_json.get('name'),
        extId=halt_json.get('extId') or halt_json.get('bahnhofsInfoId'),
        abfahrtsZeitpunkt=_parse_iso(halt_json.get('abfahrtsZeitpunkt')),
        ankunftsZeitpunkt=_parse_iso(halt_json.get('ankunftsZeitpunkt')),
        gleis=halt_json.get('gleis'),
        routeIdx=halt_json.get('routeIdx')
    )


def parse_connection(conn_json: Dict[str, Any]) -> Connection:
    trip_id = conn_json.get('tripId') or conn_json.get('id')
    sections: List[ConnectionSection] = []

    for sec in conn_json.get('verbindungsAbschnitte', []):
        halte_list = [parse_halt(h) for h in sec.get('halte', [])]
        vm = sec.get('verkehrsmittel') or {}
        sec_obj = ConnectionSection(
            idx=sec.get('idx', 0),
            journeyId=sec.get('journeyId'),
            verkehrsmittel_name=(vm.get('name') if isinstance(vm, dict) else None),
            verkehrsmittel_nummer=(vm.get('nummer') if isinstance(vm, dict) else None),
            halte=halte_list,
            abfahrtsZeitpunkt=_parse_iso(sec.get('abfahrtsZeitpunkt')),
            ankunftsZeitpunkt=_parse_iso(sec.get('ankunftsZeitpunkt')),
            dauer_seconds=sec.get('abschnittsDauer') or sec.get('ezAbschnittsDauerInSeconds')
        )
        sections.append(sec_obj)

    preis_json = conn_json.get('angebotsPreis')
    preis = None
    if preis_json and isinstance(preis_json, dict) and 'betrag' in preis_json:
        preis = Price(betrag=preis_json.get('betrag'), waehrung=preis_json.get('waehrung', 'EUR'))

    return Connection(
        tripId=trip_id,
        sections=sections,
        dauer_seconds=conn_json.get('verbindungsDauerInSeconds', 0),
        preis=preis,
        umstiege=conn_json.get('umstiegsAnzahl', 0)
    )


# --------------------------- DB API client -------------------------------

class DBAPIClient:
    def __init__(self, base_url: str = "https://www.bahn.de", session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip('/')
        self.session = session or requests.Session()
        self.session.headers.update({
            'User-Agent': 'python-requests/DBAPIClient',
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
            'Referer': 'https://www.bahn.de/'
        })

    def search_stations(self, query: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        params = {'suchbegriff': query, 'typ': 'ALL', 'limit': limit}
        try:
            r = self.session.get(f"{self.base_url}/web/api/reiseloesung/orte", params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"Fehler bei Stationssuche: {e}")
            return None

    def search_connections(self, from_station: str, to_station: str, departure_time: Optional[datetime] = None, arrival_search: bool = False) -> Optional[Dict[str, Any]]:
        if departure_time is None:
            departure_time = datetime.now()
        payload = {
            "abfahrtsHalt": from_station,
            "anfrageZeitpunkt": departure_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ankunftsHalt": to_station,
            "ankunftSuche": "ANKUNFT" if arrival_search else "ABFAHRT",
            "klasse": "KLASSE_2",
            "produktgattungen": ["ICE","EC_IC","IR","REGIONAL","SBAHN","BUS","SCHIFF","UBAHN","TRAM","ANRUFPFLICHTIG"],
            "reisende": [{
                "typ": "ERWACHSENER",
                "ermaessigungen": [{"art": "KEINE_ERMAESSIGUNG", "klasse": "KLASSENLOS"}],
                "alter": [],
                "anzahl": 1
            }],
            "schnelleVerbindungen": True,
            "sitzplatzOnly": False,
            "bikeCarriage": False
        }
        try:
            r = self.session.post(f"{self.base_url}/web/api/angebote/fahrplan", json=payload, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"Fehler bei Verbindungs-Anfrage: {e}")
            raise


# ------------------------- Helper operations -----------------------------


def find_connection_in_results(results_json: Dict[str, Any], train_number: str, departure_place: Optional[str], departure_time: Optional[datetime], tolerance_minutes: int = 15) -> Optional[Dict[str, Any]]:
    """Scan API results (a dict with 'verbindungen') for a connection with given train_number,
    departure_place (name) and roughly matching departure_time (within tolerance).

    Returns the raw connection dict if found.
    """
    if not results_json:
        return None
    for conn in results_json.get('verbindungen', []):
        for sec in conn.get('verbindungsAbschnitte', []):
            vm = sec.get('verkehrsmittel') or {}
            nummer = vm.get('nummer') if isinstance(vm, dict) else None
            if nummer and str(nummer) == str(train_number):
                # check halts for matching departure place/time
                halts = sec.get('halte', [])
                for h in halts:
                    name = h.get('name')
                    ts = _parse_iso(h.get('abfahrtsZeitpunkt'))
                    # if a departure_place name is provided, prefer name-matching
                    if departure_place and name and departure_place.lower() not in name.lower():
                        continue
                    if departure_time and ts:
                        diff = abs((ts - departure_time).total_seconds()) / 60.0
                        if diff <= tolerance_minutes:
                            return conn
                    else:
                        # no departure_time provided, match by name only
                        return conn
    return None

def get_extId_for_departure_place(json: Dict[str, Any], lazy: bool=True ) -> Optional[str]:
    idx = 0
    if not lazy and json and len(json) > 0:
        print([f"{i+1}: {ort.get('name')}" for i, ort in enumerate(json)])
        in1 = input("Welche Abfahrtsstelle soll verwendet werden? (Nummer): ").strip()
        idx = int(in1) - 1 if in1 else 0
    if json and len(json) > idx:
        ort = json[idx]
        return ort.get('extId'), ort.get('id'), (ort.get('lat'), ort.get('lon'))
    
    
def get_price_for_train(client: DBAPIClient, train_number: str, src: str, dst: str, departure_time: datetime) -> Optional[Price]:
    """Call the API for src->dst at departure_time and return the price for the matching train number.
    Returns Price or None if not found.
    """
    resp = client.search_connections(src, dst, departure_time)
    if not resp:
        return None
    for conn in resp.get('verbindungen', []):
        # top-level price sometimes present
        price = conn.get('angebotsPreis')
        # check if this connection contains the train number
        for sec in conn.get('verbindungsAbschnitte', []):
            vm = sec.get('verkehrsmittel') or {}
            nummer = vm.get('nummer') if isinstance(vm, dict) else None
            if nummer and str(nummer) == str(train_number):
                if price and isinstance(price, dict) and 'betrag' in price:
                    try:
                        return Price(betrag=float(price.get('betrag')), waehrung=price.get('waehrung', 'EUR'))
                    except Exception:
                        return Price(betrag=price.get('betrag'), waehrung=price.get('waehrung', 'EUR'))
                # fallback: section-level price
                for s in conn.get('verbindungsAbschnitte', []):
                    if 'angebotsPreis' in s:
                        p = s.get('angebotsPreis')
                        if p and isinstance(p, dict) and 'betrag' in p:
                            try:
                                return Price(betrag=float(p.get('betrag')), waehrung=p.get('waehrung', 'EUR'))
                            except Exception:
                                return Price(betrag=p.get('betrag'), waehrung=p.get('waehrung', 'EUR'))
    return None


def get_stops_from_connection(raw_connection: Dict[str, Any]) -> List[Halt]:
    """Return ordered list of Halt dataclasses from a raw connection dict.
    Merges halts from all sections and orders them by routeIdx when available.
    """
    halts: List[Halt] = []
    for sec in raw_connection.get('verbindungsAbschnitte', []):
        for h in sec.get('halte', []):
            halts.append(parse_halt(h))
    # try to sort by routeIdx then by arrival/dep time
    def _key(h: Halt):
        if h.routeIdx is not None:
            return (h.routeIdx, h.abfahrtsZeitpunkt or h.ankunftsZeitpunkt or datetime.min)
        return (999999, h.abfahrtsZeitpunkt or h.ankunftsZeitpunkt or datetime.min)

    halts_sorted = sorted(halts, key=_key)
    # remove duplicates (same extId and same routeIdx)
    seen = set()
    out = []
    for h in halts_sorted:
        key = (h.extId, h.routeIdx, h.abfahrtsZeitpunkt, h.ankunftsZeitpunkt)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


# --------------------------- Dijkstra -----------------------------------

def dijkstra(graph: Dict[str, Dict[str, float]], source: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """Classic Dijkstra.
    graph: adjacency dict {node: {neighbor: weight, ...}, ...}
    Returns (distances, previous) where previous maps node->prev_node
    """
    dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, Optional[str]] = {source: None}
    pq: List[Tuple[float, str]] = [(0.0, source)]

    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float('inf')):
            continue
        for nei, w in graph.get(node, {}).items():
            nd = d + float(w)
            if nd < dist.get(nei, float('inf')):
                dist[nei] = nd
                prev[nei] = node
                heapq.heappush(pq, (nd, nei))
    return dist, prev


def reconstruct_path(prev: Dict[str, Optional[str]], start: str, goal: str) -> List[str]:
    if goal not in prev:
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        if cur == start:
            break
        cur = prev.get(cur)
    return list(reversed(path))


# --------------------------- Interactive utils ---------------------------


def choose_station_interactive(client: DBAPIClient, query: str, lazy = False) -> Optional[Dict[str, Any]]:
    stations = client.search_stations(query)
    if not stations:
        print("Keine Stationen gefunden")
        return None
    for i, s in enumerate(stations):
        name = s.get('name') or s.get('label') or s.get('title')
        sid = s.get('id') or s.get('extId')
        if not lazy:
            print(f"{i+1}: {name} (id={sid})")
    idx = '1'
    if not lazy:
        idx = input("Wähle die Nummer der Station (oder Enter für 1): ")
    try:
        idxn = int(idx) - 1 if idx.strip() else 0
        return stations[idxn]
    except Exception:
        print("Ungültige Auswahl, nehme erste")
        return stations[0]


def _find_stop_index(stops: List[Halt], station_obj: Dict[str, Any]) -> Optional[int]:
    """Try to find index of a stop inside stops by comparing id or name."""
    if not station_obj:
        return None
    sid = station_obj.get('id') or station_obj.get('extId') or station_obj.get('label')
    sname = (station_obj.get('name') or station_obj.get('label') or '').lower()
    for idx, h in enumerate(stops):
        if not h:
            continue
        # prefer exact id match
        if sid and h.id and sid == h.id:
            return idx
        # try extId match
        if sid and h.extId and sid == h.extId:
            return idx
        # try name contains
        if sname and h.name and sname in h.name.lower():
            return idx
    return None


def build_segment_price_graph(client: DBAPIClient, train_no: str, stops: List[Halt], start_idx: int, end_idx: int, pause_seconds: float = 2.0) -> Dict[str, Dict[str, float]]:
    """For each stop i in [start_idx, end_idx) and each j>i up to end_idx, query price for i->j on the same train.
    Returns an adjacency dict {stop_id: {later_stop_id: price, ...}, ...}
    """
    graph: Dict[str, Dict[str, float]] = {}
    n = len(stops)
    if start_idx < 0 or end_idx >= n or start_idx >= end_idx:
        return graph

    for i in range(start_idx, end_idx):
        si = stops[i]
        sid = si.id or si.extId or f"{i}_{si.name}"
        graph.setdefault(sid, {})
        # prefer a departure time for the API call
        dep_time = si.abfahrtsZeitpunkt or si.ankunftsZeitpunkt or datetime.now()
        for j in range(i+1, end_idx+1):
            sj = stops[j]
            tid = sj.id or sj.extId or f"{j}_{sj.name}"
            # call price query
            try:
                time.sleep(pause_seconds)  # be polite
                p = get_price_for_train(client, train_no, sid, tid, dep_time)
                print(f"Preis für {si.name} -> {sj.name}: {p.betrag if p else 'keine Preisinformation'}")
                if p is not None:
                    # ensure numeric
                    graph[sid][tid] = float(p.betrag)
            except Exception as e:
                # don't explode on single failures
                print(f"Warnung: Preisabfrage fehlgeschlagen für {si.name} -> {sj.name}: {e}")
                continue
    return graph

def call_with_backoff(func, *args, max_retries=7, base_delay=2.0, exceptions=(requests.HTTPError, requests.ConnectionError), **kwargs):
    """Call a function with retries and exponential backoff.

    Args:
        func: The callable to execute (pass the function object, NOT the result of calling it).
        *args: Positional arguments for func.
        max_retries: Maximum number of retries before failing.
        base_delay: Initial delay in seconds.
        exceptions: Tuple of exception types to catch and retry.
        **kwargs: Keyword arguments for func.

    Returns:
        The return value of func.

    Raises:
        RuntimeError: If all retries fail.
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Attempt {attempt+1} for {func.__name__}...")
            result = func(*args, **kwargs)
            # If func returns a requests.Response, check status
            if isinstance(result, requests.Response):
                if result.status_code == 429:
                    # Explicitly raise so it's caught and triggers backoff
                    raise requests.HTTPError("Rate limit hit", response=result)
                result.raise_for_status()
            return result
        except Exception as e:
            wait_time = base_delay * (2 ** attempt)
            jitter = random.uniform(0, 1)
            print(f"Attempt {attempt+1} failed: {e}. Waiting {wait_time + jitter:.1f}s before retry...")
            time.sleep(wait_time + jitter)
    raise RuntimeError(f"Function {func.__name__} failed after {max_retries} retries")

def next_workday_noon(holidays=None):
    """
    Get datetime of 12:00 on the next workday (Mon–Fri, not in holidays),
    or today if it's before noon.
    `holidays` should be a set of date objects.
    """
    holidays = holidays or set()
    now = datetime.now()
    
    # base candidate = today at 12:00
    candidate = now.replace(hour=12, minute=0, second=0, microsecond=0)
    
    # If we are already past 12:00 today, move to tomorrow
    if now >= candidate:
        candidate += timedelta(days=1)
    
    # Skip weekends and holidays
    while candidate.weekday() >= 5 or candidate.date() in holidays:
        candidate += timedelta(days=1)
    
    return candidate


# ---------------------------- Next departure finder API -----------------


def get_abfahrten(ort_ext_id, ort_id, date_time, train_types=("ICE", "EC_IC", "IR"), max_vias=8):
    """
    Calls the DB 'abfahrten' API for a given station.
    
    ort_ext_id: e.g. '8000199'
    ort_id:     full A=1@O=... string
    date_time:  datetime object (local station time)
    train_types: iterable of transport types like ("ICE", "EC_IC")
    """
    url = "https://www.bahn.de/web/api/reiseloesung/abfahrten"
    
    params = {
        "datum": date_time.strftime("%Y-%m-%d"),
        "zeit": date_time.strftime("%H:%M:%S"),
        "ortExtId": ort_ext_id,
        "ortId": ort_id,
        "mitVias": "true",
        "maxVias": str(max_vias)
    }
    
    for t in train_types:
        params.setdefault("verkehrsmittel[]", [])
        params["verkehrsmittel[]"].append(t)
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def is_in_germany(place):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place,
        "format": "json",
        "addressdetails": 1,
        "limit": 1
    }
    headers = {"User-Agent": "my-app"}
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    data = r.json()
    if not data:
        return False
    country = data[0]["address"].get("country_code", "").lower()
    return country == "de"



# --------------------------- Main interactive flow ----------------------

if __name__ == '__main__':
    client = DBAPIClient()

    # 1) User input
    src_query = input("Startort (z.B. 'München Hbf'): ").strip()
    if src_query != '':
        dst_query = input("Zielort (z.B. 'Berlin Hbf'): ").strip()
        dt_str = input("Abfahrtszeit (YYYY-MM-DD HH:MM) — leer = jetzt: ").strip()
        train_no = input("Zugnummer (z.B. 'ICE 1102' oder nur '1102'): ").strip()
        

        if not src_query or not dst_query or not train_no:
            print("Bitte Start, Ziel und Zugnummer angeben.")
            raise SystemExit(1)

        # parse datetime
        if not dt_str:
            departure_time = datetime.now()
        else:
            try:
                departure_time = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            except Exception:
                print("Ungültiges Datumsformat. Nutze aktuelle Zeit.")
                departure_time = datetime.now()

        # 2) Station lookup (interactive chooser)
        src_station = choose_station_interactive(client, src_query)
        dst_station = choose_station_interactive(client, dst_query)
        if not src_station or not dst_station:
            print("Stationen konnten nicht aufgelöst werden.")
            raise SystemExit(1)

        src_id = src_station.get('id') or src_station.get('extId')
        dst_id = dst_station.get('id') or dst_station.get('extId')
        src_name = src_station.get('name') or src_station.get('label') or ''
        dst_name = dst_station.get('name') or dst_station.get('label') or ''

        # 3) Query connections for the route and find the specific connection by train number
        print("Suche Verbindungen vom API...")
        resp = client.search_connections(src_id, dst_id, departure_time)
        if not resp:
            print("Fehler: Keine Verbindungen vom API erhalten.")
            raise SystemExit(1)

        # ensure train_no is just the number part if user typed 'ICE 1102'
        train_no_clean = ''.join(ch for ch in train_no if ch.isdigit()) or train_no

        raw_conn = find_connection_in_results(resp, train_no_clean, src_name, departure_time)
        if not raw_conn:
            print("Keine passende Verbindung mit diesem Zug gefunden. Versuch weitere Ergebnisse...")
            # try broader search (no to_station) — expensive / optional
            # As a fallback, try searching from src to itself to get any connections nearby
            raw_conn = find_connection_in_results(resp, train_no_clean, None, departure_time)

        if not raw_conn:
            print("Keine Verbindung gefunden, die dem angegebenen Zug entspricht.")
            raise SystemExit(1)

        conn_parsed = parse_connection(raw_conn)
        print(f"Gefundene Verbindung: tripId={conn_parsed.tripId}, Dauer(s)={conn_parsed.dauer_seconds}")

        # 4) Extract ordered stops
        stops = get_stops_from_connection(raw_conn)
        if not stops:
            print("Keine Haltestellen in der Verbindung gefunden.")
            raise SystemExit(1)

        # 5) Find indices of the start and end station inside the stops list
        start_idx = _find_stop_index(stops, src_station)
        end_idx = _find_stop_index(stops, dst_station)
        if start_idx is None or end_idx is None:
            print("Start- oder Zielhaltestelle nicht in den Halten der gefundenen Verbindung gefunden.")
            # show stops for debugging
            print("Halte der Verbindung:")
            for i, s in enumerate(stops):
                print(f"{i}: {s.name} (id={s.id})")
            raise SystemExit(1)

        if start_idx >= end_idx:
            print("Der gewählte Zug fährt nicht vom Start zum Ziel in dieser Reihenfolge (Start kommt nach Ziel).")
            raise SystemExit(1)

        print(f"Train fährt von '{stops[start_idx].name}' (idx={start_idx}) nach '{stops[end_idx].name}' (idx={end_idx})")

        # 6) Price queries for every forward segment (start -> any later stop up to destination)
        print("Frage Preise für Segmente ab... (kann einige Zeit dauern, vermeide zu viele Anfragen)")
        graph = build_segment_price_graph(client, train_no_clean, stops, start_idx, end_idx, pause_seconds=2.0)

        if not graph:
            print("Es konnten keine Preisinformationen abgerufen werden.")
            raise SystemExit(1)

        # 7) Run Dijkstra from start to destination
        sid_start = (stops[start_idx].id or stops[start_idx].extId or f"{start_idx}_{stops[start_idx].name}")
        sid_end = (stops[end_idx].id or stops[end_idx].extId or f"{end_idx}_{stops[end_idx].name}")

        dist, prev = dijkstra(graph, sid_start)
        if sid_end not in dist:
            print("Kein Pfad mit verfügbaren Preis-Segmenten gefunden, das zum Ziel führt.")
            raise SystemExit(1)

        path_ids = reconstruct_path(prev, sid_start, sid_end)

        # 8) Validate path covers only forward indices and belongs to this connection
        # build id->index map
        id_to_index: Dict[str, int] = {}
        for idx, h in enumerate(stops):
            hid = h.id or h.extId or f"{idx}_{h.name}"
            id_to_index[hid] = idx

        # check monotonic increasing
        valid = True
        prev_idx = -1
        for pid in path_ids:
            if pid not in id_to_index:
                valid = False
                break
            if id_to_index[pid] <= prev_idx:
                valid = False
                break
            prev_idx = id_to_index[pid]

        if not valid:
            print("Der gefundene Pfad ist ungültig für die Halte-Reihenfolge der Verbindung.")
            raise SystemExit(1)

        # 9) Present result
        total_cost = dist[sid_end]
        print("Günstigster Pfad (keine Zugwechsel, gleicher Zug):")
        for a, b in zip(path_ids, path_ids[1:]):
            ia = id_to_index[a]
            ib = id_to_index[b]
            name_a = stops[ia].name
            name_b = stops[ib].name
            price = graph.get(a, {}).get(b)
            print(f"  {name_a} -> {name_b}: {price} { 'EUR' if price is not None else '' }")
        print(f"Gesamtkosten: {total_cost} EUR")

        # quick summary of times
        print("Zeitliche Übersicht der Haltepunkte im Zug:")
        for idx in range(start_idx, end_idx+1):
            h = stops[idx]
            t = h.abfahrtsZeitpunkt or h.ankunftsZeitpunkt
            print(f"  {idx}: {h.name} @ {t}")

        print("Fertig.")


client = DBAPIClient()
def get_all_ice_ic_stations():
    major_hubs = ["Berlin Hbf", "München Hbf", "Flensburg Hbf", "Frankfurt(Main) Hbf", "Köln Hbf", "Stuttgart Hbf", "Westerland (Sylt) Hbf", "Norddeich Mole", "Koblenz Hbf", "Saarbrücken Hbf", "Freiburg (Breisgau) Hbf",
                "Tübingen Hbf", "Friedrichshafen Stadt", "Seefeld", "Berchtesgaden Hbf", "Passau Hbf", "Gera Hbf", "Dresden Hbf", "Ostseebad Binz", "Kiel Hbf", "Lübeck Hbf", "Warnemünde Strand", "Frankfurt (Oder) Hbf"]
    major_hubs = [choose_station_interactive(client, h, lazy=True) for h in major_hubs]
    all_stations = {}

    for i,hub in enumerate(major_hubs):
        # hub_info = choose_station_interactive(client, hub, lazy = True)
        if not hub:
            continue
        for j,other in enumerate(major_hubs):
            if i >= j:
                continue
            if not other:
                continue
            time.sleep(random.uniform(0, 1))  # avoid hammering the API
            resp = call_with_backoff(client.search_connections, hub['id'], other['id'], next_workday_noon(), max_retries=7, base_delay=2.0)
            if not resp:
                print("too fast for api, but how?")
                exit(1)
            for conn in resp.get('verbindungen', []):
                for sec in conn.get('verbindungsAbschnitte', []):
                    vm = sec.get('verkehrsmittel') or {}
                    if vm.get('produktGattung') in ("ICE", "EC_IC", "IR"):
                        for h in sec.get('halte', []):
                            all_stations[h['id']] = h['name']

    # Save as file
    with open(local_folder + '/db/ice_ic_stations.json', 'w', encoding='utf-8') as f:
        json.dump(all_stations, f, indent=2, ensure_ascii=False)
    return all_stations

# sts = get_all_ice_ic_stations()

import networkx as nx

holidays = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 6),   # Epiphany (BW, BY, ST)
    date(2025, 3, 8),   # Int. Women's Day (BE, MV)
    date(2025, 4, 18),  # Good Friday
    date(2025, 4, 20),  # Easter Sunday (BB)
    date(2025, 4, 21),  # Easter Monday
    date(2025, 5, 1),   # Labour Day
    date(2025, 5, 8),   # 80th End of WWII Anniversary (BE)
    date(2025, 5, 29),  # Ascension Day
    date(2025, 6, 8),   # Whit Sunday (BB)
    date(2025, 6, 9),   # Whit Monday
    date(2025, 6, 19),  # Corpus Christi (BW, BY, HE, NW, RP, SL, SN, TH)
    date(2025, 8, 15),  # Assumption Day (BY, SL)
    date(2025, 9, 20),  # Children's Day (TH)
    date(2025, 10, 3),  # Day of German Unity
    date(2025, 10, 31), # Reformation Day
    date(2025, 11, 1),  # All Saints' Day (BW, BY, NW, RP, SL)
    date(2025, 11, 19), # Repentance Day (SN)
    date(2025, 12, 25), # Christmas Day
    date(2025, 12, 26)
    }



def get_train_edges(G, all_stations, all_st2):
    
    not_found_nodes = set()
    # G = nx.DiGraph()
    # with open(local_folder + '/db/ice_ic_stations.json', 'r', encoding='utf-8') as f:
    #     all_stations = json.load(f)
    diff_entries = {k: v for k, v in all_st2.items() if k not in all_stations}

    for i, (st1, name) in enumerate(diff_entries.items()):
        print(f"Processing station {i+1}/{len(diff_entries)}: {name} ({st1}) ({len(G.nodes())} nodes, {round((i+1) / len(diff_entries), 6) * 100} %)")
        extId, id, coords = get_extId_for_departure_place(client.search_stations(name), lazy=True)
        G.add_node(name, extId=extId, id=id, coords=coords)
        for hour in range(0, 24, 1):
            time.sleep(random.uniform(0, 1))  # avoid hammering the API
            try:
                resp = call_with_backoff(get_abfahrten, extId, id, next_workday_noon() + timedelta(hours=hour), max_retries=7, base_delay=2.0)
                if not resp:
                    print(name, f" time:{(12 + hour) % 24} o'clock missing, continuing...")
                    continue
                for train in resp.get('entries', []):
                    # print(train)
                    
                    if train.get('ueber') and len(train.get('ueber', [])) > 0 and train.get('verkehrmittel', {}).get('produktGattung') in ("ICE", "EC_IC", "IR"):
                        n1 = None

                        for via in train.get('ueber', []):
                            if via == name:
                                n1 = via
                                continue
                            if via in all_st2.values() and n1 in all_st2.values() and (is_in_germany(n1) and is_in_germany(via)):
                                # print(f"Adding edge from {n1} to {via} for train {train.get('verkehrmittel', {}).get('name')}")
                                G.add_edge(n1, via, t=[])
                                G.edges[n1,via]['t'].append(train.get('verkehrmittel', {}).get('name'))
                                n1 = via
                            else:
                                if via not in all_st2.values() and via not in not_found_nodes:
                                    print(f"Fehler: Konnte Knoten für {via} nicht finden.")
                                    not_found_nodes.add(via)
                                elif n1 not in all_st2.values() and n1 not in not_found_nodes:
                                    print(f"Fehler: Konnte Knoten für {n1} nicht finden.")                                        
                                    not_found_nodes.add(n1)
                                    
            except Exception as e:
                print(f"Error processing:\n{e}")
                continue
    # Save the graph to a file
    nx.write_gpickle(G, local_folder + '/db/ice_ic_graph2.gpickle')
    return not_found_nodes

# all_st2 = {}
# with open(local_folder + '/db/ice_ic_stations.json', 'r', encoding='utf-8') as f:
#     all_stations = json.load(f)
#     with open(local_folder + '/db/ice_ic_stations_2.json', 'r', encoding='utf-8') as f3:
#         all_st2 = json.load(f3)
#     nfn = get_train_edges(nx.read_gpickle(local_folder + '/db/ice_ic_graph.gpickle'), all_stations, all_st2)
#     print(nfn)


nfng = ['Dagebüll Kirche', 'Fischen', 'Bad Schandau', 'Lübbenau(Spreewald)', 'Langenhagen Mitte', 'Kempten(Allgäu)Hbf', 'Oberndorf(Neckar)', 'Geilenkirchen', 'Kaufbeuren', 'Lippstadt', 'Altenbeken', 'Lindau-Reutin', 'Rudolstadt(Thür)', 'Müllheim im Markgräflerland', 'Gäufelden', 'Naumburg(Saale)Hbf', 'Sulz(Neckar)', 'Horb', 'München Hbf Gl.27-36', 'Bad Bentheim', 'Saalfeld(Saale)', 'Ergenzingen', 'Kreuztal', 'Landstuhl', 'Neu-Ulm', 'Paderborn Hbf', 'Dillenburg', 'Wittlich Hbf', 'Werdohl', 'Bad Nauheim', 'Lichtenfels', 'Warburg(Westf)', 'Cochem(Mosel)', 'Chemnitz Hbf', 'Ludwigsstadt', 'Konstanz', 'Rheydt Hbf', 'Emden Außenhafen', 'Steinach(bei Rothenburg ob der Tauber)', 'Siegen Hbf', 'Finnentrop', 'Aachen Hbf', 'Sonthofen', 'Krefeld Hbf', 'Treis-Karden', 'Viersen', 'Kobern-Gondorf', 'Witten Hbf', 'Schorndorf', 'Buchloe', 'Deezbüll', 'Boppard Hbf', 'Immenstadt', 'Berlin Ostkreuz', 'Lindau-Insel', 'Lübben(Spreewald)', 'Siegen-Weidenau', 'Erkelenz', 'Düren', 'Köln/Bonn Flughafen', 'Mönchengladbach Hbf', 'Soest', 'Altena(Westf)', 'Friedrichshafen Stadt', 'Weißenfels', 'Biberach(Riß)', 'Böblingen', 'Frankfurt(M) Flughafen Regionalbf', 'Bullay(DB)', 'Bondorf(b Herrenberg)', 'Königs Wusterhausen', 'Weil am Rhein', 'Oberstdorf', 'Ravensburg', 'Herrenberg', 'Wiesbaden Hbf', 'Cottbus Hbf', 'Rottweil', 'Memmingen', 'Pinneberg', 'Ludwigsburg', 'Freiberg(Sachs)', 'Lahr(Schwarzw)', 'Dagebüll Mole', 'Mülheim(Ruhr)Hbf', 'Neuss Hbf', 'Plettenberg', 'Jena Paradies', 'Letmathe', 'Kronach', 'Köln Messe/Deutz', 'Ludwigshafen(Rh)Hbf', 'Wetzlar', 'Niebüll neg', 'Eutingen im Gäu']




def get_german_cities(nfn):
    nfng = set()
    for city in nfn:
        try:
            print(f"{city}")
            if is_in_germany(city):
                nfng.add(city)
        except Exception as e:
            print(f'exception: {e}')
    print(nfng)
    return nfng
# get_german_cities(['Lennestadt-Grevenbrück', 'Stuttgart-Vaihingen', 'Tuttlingen', 'Engen', 'Singen(Hohentwiel)', 'Herzogenrath', 'Radolfzell', 'Köln-Ehrenfeld', 'Lennestadt-Altenhundem', 'Spaichingen'])
nfng = ['Lennestadt-Grevenbrück', 'Stuttgart-Vaihingen', 'Tuttlingen', 'Engen', 'Singen(Hohentwiel)', 'Herzogenrath', 'Radolfzell', 'Köln-Ehrenfeld', 'Lennestadt-Altenhundem', 'Spaichingen']
def merge_cities():
    f1 = {}
    with open('./just_for_fun/db/ice_ic_stations.json', 'r', encoding='utf-8') as f:
        f1 = json.loads(f.read())
        # print(f1)

        for city in nfng:
            time.sleep(random.uniform(0,1))
            resp = client.search_stations(city, limit=1)[0]
            prod = resp.get('products')
            if 'EC_IC' in prod or 'ICE' in prod or 'IR' in prod:
                h_id = resp.get('id')
                if h_id not in f1:
                    print(f'added {city}')
                    f1[h_id] = city
    with open('./just_for_fun/db/ice_ic_stations_2.json', 'w', encoding='utf-8') as f2:
        json.dump(f1, f2, indent= 2, ensure_ascii=False)

# merge_cities()







G = nx.read_gpickle(local_folder + '/db/ice_ic_graph2.gpickle')
G.remove_node('Pinneberg') # no connections for pinneberg on an example workday, checked each hour

# G.add_node('test', extId='test_ext', id='test_id', coords=(0, 0))
# G.add_node('test2', extId='test_ext2', id='test_id2', coords=(1, 1))
# G.add_edge('test', 'test2', t=['ICE 1234'])

# print(G.edges())
# print(f"DB nodes with 0 edges: {len([n for n, d in G.in_degree() if d == 0])}")
import matplotlib.pyplot as plt

# mapping = {node: node[1] for node in G.nodes()}
# G = nx.relabel_nodes(G, mapping)



pos = nx.spring_layout(G, seed=42)  # positions for all nodes
nx.draw(G, pos, node_size=300, font_size=8, with_labels=True)
plt.show()


