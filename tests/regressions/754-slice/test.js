export const run = {
  run() {
    // Even simpler: just print PASS or FAIL based on slice result.
    const s = "hello";
    const got = s.slice(0, 3);
    if (got === "hel") {
      console.log("PASS slice=" + JSON.stringify(got));
    } else {
      console.log("FAIL slice=" + JSON.stringify(got));
    }
  },
};
