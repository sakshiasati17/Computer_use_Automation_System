from functools import wraps
import time

from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "legacy-credit-union-secret-key-2005"

SESSION_TIMEOUT_SECONDS = 5 * 60

USERNAME = "admin"
PASSWORD = "admin123"

MEMBERS = {
    "M-1001": {
        "name": "Jane Smith",
        "account": "ACT-10011",
        "savings": "12,450.00",
        "checking": "3,200.50",
    },
    "M-1002": {
        "name": "Robert Johnson",
        "account": "ACT-10022",
        "savings": "8,900.75",
        "checking": "1,500.00",
    },
    "M-1003": {
        "name": "Maria Garcia",
        "account": "ACT-10033",
        "savings": "25,100.00",
        "checking": "7,800.25",
    },
    "M-1004": {
        "name": "David Lee",
        "account": "ACT-10044",
        "savings": "4,200.00",
        "checking": "950.00",
    },
    "M-1005": {
        "name": "Sarah Williams",
        "account": "ACT-10055",
        "savings": "15,750.50",
        "checking": "4,100.00",
    },
}


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        last_activity = session.get("last_activity", 0)
        if time.time() - last_activity > SESSION_TIMEOUT_SECONDS:
            session.clear()
            return redirect(url_for("login", expired=1))
        session["last_activity"] = time.time()
        return f(*args, **kwargs)

    return wrapper


@app.route("/")
def index():
    if session.get("logged_in"):
        return redirect(url_for("search"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == USERNAME and password == PASSWORD:
            session.clear()
            session["logged_in"] = True
            session["username"] = username
            session["last_activity"] = time.time()
            return redirect(url_for("search"))
        error = "Invalid credentials"
    elif request.args.get("expired"):
        error = "Session expired. Please log in again."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/search", methods=["GET", "POST"])
@login_required
def search():
    error = None
    if request.method == "POST":
        member_id = request.form.get("member_id", "").strip()
        if member_id in MEMBERS:
            return redirect(url_for("member_detail", member_id=member_id))
        error = "Member not found"
    return render_template("search.html", error=error)


@app.route("/member/<member_id>")
@login_required
def member_detail(member_id):
    if member_id not in MEMBERS:
        return redirect(url_for("search"))
    return render_template("member_detail.html", member_id=member_id)


@app.route("/member/<member_id>/frame")
@login_required
def member_frame(member_id):
    member = MEMBERS.get(member_id)
    if not member:
        return redirect(url_for("search"))
    return render_template("member_frame.html", member_id=member_id, member=member)


@app.route("/member/<member_id>/new-account", methods=["GET", "POST"])
@login_required
def new_account(member_id):
    member = MEMBERS.get(member_id)
    if not member:
        return redirect(url_for("search"))

    dialog = request.args.get("dialog") == "true"
    if dialog:
        action_url = url_for("new_account", member_id=member_id, dialog="true")
    else:
        action_url = url_for("new_account", member_id=member_id)

    error = None
    if request.method == "POST":
        account_type = request.form.get("account_type", "Savings")
        deposit_raw = request.form.get("deposit", "")
        try:
            deposit = float(deposit_raw)
        except ValueError:
            deposit = -1
        if deposit < 25:
            error = "Minimum initial deposit is $25.00"
        else:
            session[f"new_account_{member_id}"] = {
                "account_type": account_type,
                "deposit": "{:,.2f}".format(deposit),
            }
            return redirect(url_for("confirmation", member_id=member_id))

    return render_template(
        "new_account.html",
        member_id=member_id,
        member=member,
        error=error,
        dialog=dialog,
        action_url=action_url,
    )


@app.route("/member/<member_id>/confirmation")
@login_required
def confirmation(member_id):
    member = MEMBERS.get(member_id)
    if not member:
        return redirect(url_for("search"))
    details = session.get(f"new_account_{member_id}")
    if not details:
        return redirect(url_for("member_detail", member_id=member_id))
    return render_template(
        "confirmation.html", member_id=member_id, member=member, details=details
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
