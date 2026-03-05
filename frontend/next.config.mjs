import path from "node:path";

/** @type {import('next').NextConfig} */
const nextConfig = {
  outputFileTracingRoot: path.join(import.meta.dirname, ".."),
};

export default nextConfig;
