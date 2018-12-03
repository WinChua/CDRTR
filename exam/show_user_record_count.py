import re
import os
import glob
import sys
sys.path.insert(0, "..")
from CDRTR.utils import pkload, pkdump
from CDRTR.preprocess.utils import readJson


def main(datadir):
    # datadir = sys.argv[1]
    if os.path.exists(os.path.join(datadir, "static_res.pk")):
        return pkload(os.path.join(datadir, "static_res.pk"))
    pattern = re.compile("reviews_(.*?)_5.json")
    jsondir = os.path.join(datadir, "preprocess", "transform")
    src, tgt = pattern.findall("\n".join(os.listdir(jsondir)))
    cold = os.path.join(datadir, "preprocess", "cold")

    def getUser(domain, overuser=None):
        domainpattern = glob.glob(os.path.join(cold, "*%s*.pk" % domain))
        domainfile = list(domainpattern)[0]
        domainuser = pkload(domainfile)

        domainjsonpattern = glob.glob(os.path.join(jsondir, "*%s*.json" % domain))

        if domain == "overlap":
            return domainuser

        domainUC = 0
        domainOUC = 0
        domainuser = set(domainuser)
        overuser = set(overuser)
        for record in readJson(domainjsonpattern[0]):
            if record["reviewerID"] in domainuser:
                domainUC += 1
            if record["reviewerID"] in overuser:
                domainOUC += 1

        return domainuser, domainUC, domainOUC

    overuser = getUser("overlap")
    srcuser, srcUC, srcOUC = getUser(src, overuser)
    tgtuser, tgtUC, tgtOUC = getUser(tgt, overuser)

    print datadir, "done"
    static_res = {"Domain": [src, tgt],
                "User": {"overlap": len(overuser), src: len(srcuser), tgt: len(tgtuser)},
                "Record": {"srcUC": srcUC, "srcOUC": srcOUC, "tgtUC": tgtUC, "tgtOUC": tgtOUC}}

    pkdump(static_res, os.path.join(datadir, "static_res.pk"))
    return static_res

Usage = '''<Usage %s data1 data2 data3 ...>

e.g:
    python show_user_record_count.py `ls -F | grep / | grep -v log | cut -d/ -f1`'''

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print Usage % sys.argv[0]
        sys.exit(1)
    from multiprocessing.pool import Pool
    from prettytable import PrettyTable
    pool = Pool(10)
    results = pool.map(main, sys.argv[1:])
    usertable = PrettyTable(["domains", "srcUserCount", "tgtUserCount", "overlapCount"])
    recdtable = PrettyTable(["domains", "srcURcount", "srcOverRcount", "tgtURcount", "tgtOverRcount"])
    for res in results:
        src, tgt = res["Domain"]
        u = res["User"]
        r = res["Record"]
        usertable.add_row([src+"/"+tgt, u[src], u[tgt], u["overlap"]])
        recdtable.add_row([src+"/"+tgt, r["srcUC"], r["srcOUC"], r["tgtUC"], r["tgtOUC"]])

    print(usertable)
    print(recdtable)

