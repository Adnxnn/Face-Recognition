import { NextResponse } from 'next/server';
export function GET(){return NextResponse.json({ready:Boolean(process.env.FACE_API_URL&&process.env.FACE_API_KEY)});}
