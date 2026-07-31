Bun.serve({
  port: process.env.PORT,
  fetch() {
    return Response.redirect("https://nsbuitrago.github.io/rma-kinetics-app", 301);
  },
});
