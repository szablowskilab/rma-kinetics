from oven/bun:1

COPY app/redirect.ts .
CMD ["bun", "redirect.ts"]
