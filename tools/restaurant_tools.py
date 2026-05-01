"""
Restaurant Operations Tools (MCP-Style Simulation)
====================================================
These functions simulate backend / MCP-server responses.  Each function
behaves as if it were calling a real external service: it accepts typed
parameters, performs validation, and returns structured JSON-like
responses with status codes.

In a production environment these would be HTTP calls to a reservation
micro-service, POS system, or MCP server.  The simulation layer is
deliberately realistic so the agent learns correct tool-calling
patterns that transfer directly to real integrations.
"""

import json
import random
from datetime import datetime, timedelta
from langchain.tools import tool


# ─── Simulated Database ──────────────────────────────────────────

BRANCHES = ["downtown", "midtown", "brooklyn"]

# Simulated table availability (realistic randomized state)
def _generate_availability(branch: str, date: str, time: str) -> dict:
    """Deterministic-ish availability based on branch + date + time."""
    # Seed with the inputs so same query → same answer within a session
    seed = hash(f"{branch}-{date}-{time}") % 10000
    rng = random.Random(seed)

    total_tables = {"downtown": 16, "midtown": 24, "brooklyn": 12}.get(branch, 16)
    # Weekends and evenings are busier
    is_weekend = False
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        is_weekend = dt.weekday() >= 5
    except ValueError:
        pass

    hour = 18  # default
    try:
        hour = int(time.split(":")[0])
    except (ValueError, IndexError):
        pass

    busy_factor = 0.8 if (is_weekend or 18 <= hour <= 21) else 0.5
    occupied = int(total_tables * busy_factor) + rng.randint(-2, 3)
    occupied = max(0, min(occupied, total_tables))
    available = total_tables - occupied

    return {
        "branch": branch,
        "date": date,
        "time": time,
        "total_tables": total_tables,
        "available_tables": max(available, 0),
    }


# Simulated daily specials
DAILY_SPECIALS = {
    "downtown": {
        0: {"dish": "Truffle Mushroom Risotto", "price": "$19.99", "description": "Creamy Arborio rice with black truffle shavings and wild mushrooms."},
        1: {"dish": "Grilled Swordfish Steak", "price": "$27.99", "description": "Fresh swordfish with lemon-caper butter and roasted vegetables."},
        2: {"dish": "Short Rib Tagliatelle", "price": "$22.99", "description": "Braised short rib over fresh tagliatelle with red wine reduction."},
        3: {"dish": "Herb-Crusted Lamb Chops", "price": "$29.99", "description": "New Zealand lamb with rosemary jus and mashed potatoes."},
        4: {"dish": "Lobster Mac & Cheese", "price": "$24.99", "description": "Maine lobster with four-cheese sauce and truffle breadcrumbs."},
        5: {"dish": "Wagyu Beef Slider Trio", "price": "$26.99", "description": "Three wagyu sliders with caramelized onions and gruyère."},
        6: {"dish": "Seafood Paella", "price": "$28.99", "description": "Spanish rice with shrimp, mussels, clams, and chorizo."},
    },
    "midtown": {
        0: {"dish": "Pan-Seared Halibut", "price": "$26.99", "description": "Atlantic halibut with saffron beurre blanc and asparagus."},
        1: {"dish": "Veal Milanese", "price": "$25.99", "description": "Breaded veal cutlet with arugula, cherry tomatoes, and Parmesan."},
        2: {"dish": "Duck Confit", "price": "$24.99", "description": "Slow-cooked duck leg with white bean cassoulet."},
        3: {"dish": "Beef Wellington (Individual)", "price": "$32.99", "description": "Filet mignon wrapped in puff pastry with mushroom duxelles."},
        4: {"dish": "Seared Scallops", "price": "$28.99", "description": "Day-boat scallops with cauliflower purée and brown butter."},
        5: {"dish": "Osso Buco", "price": "$27.99", "description": "Braised veal shank with gremolata and saffron risotto."},
        6: {"dish": "Chilean Sea Bass", "price": "$31.99", "description": "Miso-glazed sea bass with bok choy and jasmine rice."},
    },
    "brooklyn": {
        0: {"dish": "BBQ Brisket Plate", "price": "$21.99", "description": "12-hour smoked brisket with cornbread and pickled slaw."},
        1: {"dish": "Fried Chicken & Waffles", "price": "$18.99", "description": "Buttermilk fried chicken with maple syrup and Belgian waffles."},
        2: {"dish": "Blackened Catfish", "price": "$19.99", "description": "Cajun-spiced catfish with dirty rice and collard greens."},
        3: {"dish": "Pulled Pork Sandwich", "price": "$16.99", "description": "Slow-cooked pulled pork with tangy slaw on a brioche bun."},
        4: {"dish": "Shrimp & Grits", "price": "$20.99", "description": "Gulf shrimp over creamy stone-ground grits with andouille sausage."},
        5: {"dish": "Smash Burger Deluxe", "price": "$17.99", "description": "Double smash patties with American cheese, special sauce, and fries."},
        6: {"dish": "Veggie Buddha Bowl", "price": "$15.99", "description": "Roasted sweet potato, quinoa, avocado, chickpeas, and tahini."},
    },
}

# Simulated bookings store (in-memory)
_bookings: list[dict] = []


# ─── Tool Definitions ────────────────────────────────────────────

@tool
def check_table_availability(date: str, time: str, branch: str) -> str:
    """
    Check live table availability at a NovaBite branch.

    Parameters
    ----------
    date : str
        The date in YYYY-MM-DD format (e.g. '2025-06-15').
    time : str
        The time in HH:MM format, 24-hour (e.g. '19:00').
    branch : str
        The branch name: 'downtown', 'midtown', or 'brooklyn'.

    Returns
    -------
    str  (JSON)
        Availability information including number of open tables.
    """
    branch = branch.lower().strip()
    if branch not in BRANCHES:
        return json.dumps({
            "status": "error",
            "message": f"Unknown branch '{branch}'. Valid branches: {', '.join(BRANCHES)}."
        })

    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return json.dumps({
            "status": "error",
            "message": f"Invalid date format '{date}'. Please use YYYY-MM-DD."
        })

    avail = _generate_availability(branch, date, time)
    avail["status"] = "success"
    return json.dumps(avail)


@tool
def book_table(name: str, date: str, time: str, branch: str, party_size: int = 2) -> str:
    """
    Book a table at a NovaBite branch.

    Parameters
    ----------
    name : str
        Guest name for the reservation.
    date : str
        The date in YYYY-MM-DD format.
    time : str
        The time in HH:MM format, 24-hour.
    branch : str
        The branch name: 'downtown', 'midtown', or 'brooklyn'.
    party_size : int
        Number of guests (default 2).

    Returns
    -------
    str  (JSON)
        Booking confirmation with a reference number, or an error.
    """
    branch = branch.lower().strip()
    if branch not in BRANCHES:
        return json.dumps({
            "status": "error",
            "message": f"Unknown branch '{branch}'. Valid branches: {', '.join(BRANCHES)}."
        })

    # Check availability first
    avail = _generate_availability(branch, date, time)
    if avail["available_tables"] <= 0:
        return json.dumps({
            "status": "error",
            "message": f"Sorry, no tables available at the {branch.title()} branch on {date} at {time}. Please try a different time or branch."
        })

    # Create booking
    booking_ref = f"NB-{random.randint(10000, 99999)}"
    booking = {
        "status": "success",
        "booking_reference": booking_ref,
        "name": name,
        "branch": branch.title(),
        "date": date,
        "time": time,
        "party_size": party_size,
        "message": f"Table booked successfully! Your reference number is {booking_ref}."
    }
    _bookings.append(booking)
    return json.dumps(booking)


@tool
def get_today_special(branch: str) -> str:
    """
    Get today's chef special at a NovaBite branch.

    Parameters
    ----------
    branch : str
        The branch name: 'downtown', 'midtown', or 'brooklyn'.

    Returns
    -------
    str  (JSON)
        Today's special dish with price and description.
    """
    branch = branch.lower().strip()
    if branch not in BRANCHES:
        return json.dumps({
            "status": "error",
            "message": f"Unknown branch '{branch}'. Valid branches: {', '.join(BRANCHES)}."
        })

    today = datetime.now().weekday()  # 0=Monday … 6=Sunday
    special = DAILY_SPECIALS[branch][today]

    return json.dumps({
        "status": "success",
        "branch": branch.title(),
        "day": datetime.now().strftime("%A"),
        "special": special,
    })


# List of all tools for the agent
TOOLS = [check_table_availability, book_table, get_today_special]
